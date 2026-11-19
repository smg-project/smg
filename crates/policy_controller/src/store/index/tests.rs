//! The index's cases: a lane's life with its row, and the store's apply
//! path held to the `kv-index` crate's reference index over streams with
//! holes, clears, restarts, gaps and snapshots.

use std::collections::BTreeMap;

use kv_index::{chain_prefix_hash, ApplyError, ReferenceIndexer};

use super::{
    admit, Blocks, ContentHash, KvEvent, LaneCounts, SequenceHash, StoredBlock, INDEX_WORKERS,
};
use crate::{
    store::{
        kv_events::{RankCursor, GAP_CLEAR_THRESHOLD},
        BatchFacts, BatchOutcome, KvEventSink, MemoryStore, ResyncReason, Signals, SnapshotChunk,
        WorkerFacts, WorkersStore,
    },
    view::WorkerId,
};

const MODEL: &str = "m";

/// The facts of a row whose signals ask for events.
fn publisher(key: &str, models: &[&str]) -> WorkerFacts {
    WorkerFacts {
        key: key.to_string(),
        engine: key.to_string(),
        incarnation: 1,
        models: models.iter().map(ToString::to_string).collect(),
        dp_size: 1,
        signals: Signals {
            events: true,
            ..Signals::default()
        },
        ..WorkerFacts::default()
    }
}

fn admitted(store: &MemoryStore, key: &str) -> WorkerId {
    let id = store.upsert(publisher(key, &[MODEL])).id();
    // The build this registration owes, asked: the cases read the row at
    // once.
    store.publish_now();
    id
}

/// The facts of the row of `engine` at `rank` of `ranks`.
fn ranked(engine: &str, rank: u16, ranks: u16) -> WorkerFacts {
    WorkerFacts {
        key: format!("{engine}/{rank}"),
        dp_rank: rank,
        dp_size: ranks,
        ..publisher(engine, &[MODEL])
    }
}

/// The blocks of `contents` as an engine names a chain: a block's hash is a
/// hash of its parent's hash and its own content.
fn chain(contents: &[u64]) -> Vec<StoredBlock> {
    let mut blocks: Vec<StoredBlock> = Vec::with_capacity(contents.len());
    for &content in contents {
        let seq_hash = match blocks.last() {
            Some(parent) => chain_prefix_hash(parent.seq_hash, ContentHash(content)),
            None => SequenceHash(content),
        };
        blocks.push(StoredBlock {
            seq_hash,
            content_hash: ContentHash(content),
        });
    }
    blocks
}

fn stored(parent: Option<SequenceHash>, blocks: &[StoredBlock]) -> Vec<KvEvent> {
    vec![KvEvent::Stored {
        parent,
        blocks: blocks.to_vec(),
    }]
}

fn batch(sequence: u64) -> BatchFacts {
    BatchFacts {
        sequence,
        ..BatchFacts::default()
    }
}

/// The rows holding a prefix of `request` in `model`'s index.
fn held(store: &MemoryStore, model: &str, request: &[u64]) -> BTreeMap<WorkerId, u32> {
    store.kv().overlaps(model, request).into_iter().collect()
}

#[test]
fn a_members_blocks_reach_its_models_index_and_a_removal_leaves_a_hole() {
    let store = MemoryStore::new();
    let (a, b) = (admitted(&store, "a"), admitted(&store, "b"));
    let long = chain(&[1, 2, 3, 4]);
    let fork = chain(&[1, 2, 9]);
    assert_eq!(
        store.admit(a, 0, batch(1), &stored(None, &long)),
        BatchOutcome::Applied
    );
    assert_eq!(
        store.admit(b, 0, batch(1), &stored(None, &fork)),
        BatchOutcome::Applied
    );
    assert_eq!(
        held(&store, MODEL, &[1, 2, 3, 4]),
        BTreeMap::from([(a, 4), (b, 2)])
    );
    assert_eq!(
        held(&store, MODEL, &[1, 2, 9]),
        BTreeMap::from([(a, 2), (b, 3)])
    );
    assert_eq!(
        held(&store, MODEL, &[7]),
        BTreeMap::new(),
        "nobody holds it"
    );
    assert_eq!(
        (store.indexed_blocks(a, 0), store.indexed_blocks(b, 0)),
        (4, 3)
    );
    // The third block goes: the fourth stays held and stops counting.
    let evicted = vec![KvEvent::Removed(vec![long[2].seq_hash])];
    assert_eq!(store.admit(a, 0, batch(2), &evicted), BatchOutcome::Applied);
    assert_eq!(
        held(&store, MODEL, &[1, 2, 3, 4]),
        BTreeMap::from([(a, 2), (b, 2)])
    );
    assert_eq!(store.indexed_blocks(a, 0), 3);
    // Stored again under its parent, the chain is whole again.
    let again = stored(Some(long[1].seq_hash), &long[2..3]);
    assert_eq!(store.admit(a, 0, batch(3), &again), BatchOutcome::Applied);
    assert_eq!(held(&store, MODEL, &[1, 2, 3, 4])[&a], 4);
    assert_eq!(store.lane_counts(a, 0), Some(LaneCounts::default()));
}

#[test]
fn a_row_leaves_with_its_blocks_and_its_successor_starts_empty() {
    let store = MemoryStore::new();
    let (a, b) = (admitted(&store, "a"), admitted(&store, "b"));
    let blocks = chain(&[1, 2, 3]);
    store.admit(a, 0, batch(1), &stored(None, &blocks));
    store.admit(b, 0, batch(1), &stored(None, &blocks[..1]));
    assert_eq!(
        held(&store, MODEL, &[1, 2, 3]),
        BTreeMap::from([(a, 3), (b, 1)])
    );
    // A new incarnation under the key: the old id's blocks leave with it,
    // and the model's other row keeps its own.
    let successor = store
        .upsert(WorkerFacts {
            incarnation: 2,
            ..publisher("a", &[MODEL])
        })
        .id();
    assert_ne!(successor, a);
    assert_eq!(held(&store, MODEL, &[1, 2, 3]), BTreeMap::from([(b, 1)]));
    assert_eq!(
        store.kv().held_blocks(MODEL),
        1,
        "the old id's blocks left the index"
    );
    assert_eq!(
        store.admit(a, 0, batch(2), &stored(None, &blocks)),
        BatchOutcome::Skipped,
        "a late batch of the old incarnation writes nothing"
    );
    assert_eq!(
        store.cursor(successor, 0),
        None,
        "the successor was not heard yet"
    );
    store.admit(successor, 0, batch(7), &stored(None, &blocks[..2]));
    assert_eq!(
        held(&store, MODEL, &[1, 2, 3]),
        BTreeMap::from([(successor, 2), (b, 1)])
    );
    // The row removed: its blocks go. The model's index stays while a row
    // writes it and goes with the last.
    let remove = |id| {
        let revision = WorkersStore::get(&store, id).expect("the row").revision;
        store.remove(id, revision).expect("removed");
    };
    remove(successor);
    assert_eq!(held(&store, MODEL, &[1, 2, 3]), BTreeMap::from([(b, 1)]));
    assert_eq!(store.indexed_blocks(successor, 0), 0);
    assert_eq!(store.kv().indexes(), 1);
    assert_eq!(
        store.admit(successor, 0, batch(8), &stored(None, &blocks)),
        BatchOutcome::Skipped
    );
    remove(b);
    assert_eq!(store.kv().indexes(), 0);
}

#[test]
fn a_row_is_indexed_under_its_first_model_at_its_own_rank_when_it_asks_for_events() {
    let store = MemoryStore::new();
    let blocks = chain(&[1, 2]);
    // No row: nothing is heard and nothing is written.
    let stranger = WorkerId(9);
    assert_eq!(
        store.admit(stranger, 0, batch(1), &stored(None, &blocks)),
        BatchOutcome::Skipped
    );
    assert_eq!(store.cursor(stranger, 0), None);
    // A row whose signals ask for no events, and one that names no model.
    let quiet = store
        .upsert(WorkerFacts {
            signals: Signals::default(),
            ..publisher("quiet", &[MODEL])
        })
        .id();
    let nameless = store.upsert(publisher("nameless", &[])).id();
    for id in [quiet, nameless] {
        assert_eq!(
            store.admit(id, 0, batch(1), &stored(None, &blocks)),
            BatchOutcome::Skipped
        );
    }
    assert_eq!(store.kv().indexes(), 0);
    assert_eq!(store.unindexed_rows(), 0, "neither asks for a lane");
    // Two names: the first holds the blocks.
    let two = store.upsert(publisher("two", &["first", "second"])).id();
    store.admit(two, 0, batch(1), &stored(None, &blocks));
    assert_eq!(held(&store, "first", &[1, 2]), BTreeMap::from([(two, 2)]));
    assert_eq!(held(&store, "second", &[1, 2]), BTreeMap::new());
    // A rank's row takes its own rank's batches alone.
    let ranked = store
        .upsert(WorkerFacts {
            dp_rank: 1,
            dp_size: 2,
            ..publisher("ranked", &[MODEL])
        })
        .id();
    assert_eq!(
        store.admit(ranked, 0, batch(1), &stored(None, &blocks)),
        BatchOutcome::Skipped,
        "another rank's batch"
    );
    assert_eq!(
        store.admit(ranked, 1, batch(1), &stored(None, &blocks)),
        BatchOutcome::Applied
    );
    assert_eq!(store.cursor(ranked, 1).expect("heard").applied, 1);
    // The same incarnation with the same first name keeps its blocks; under
    // another first name it starts over there.
    store.upsert(WorkerFacts {
        priority: 3,
        ..publisher("two", &["first", "second"])
    });
    assert_eq!(held(&store, "first", &[1, 2]), BTreeMap::from([(two, 2)]));
    store.upsert(publisher("two", &["second", "first"]));
    assert_eq!(held(&store, "first", &[1, 2]), BTreeMap::new());
    assert_eq!(store.indexed_blocks(two, 0), 0);
    store.admit(two, 0, batch(1), &stored(None, &blocks));
    assert_eq!(held(&store, "second", &[1, 2]), BTreeMap::from([(two, 2)]));
}

#[test]
fn a_store_under_a_parent_the_worker_does_not_hold_starts_a_chain_and_is_counted() {
    let store = MemoryStore::new();
    let a = admitted(&store, "a");
    let blocks = chain(&[1, 2, 3, 4]);
    // The first half was never heard: the second half names a parent the
    // index does not hold for this worker.
    let tail = stored(Some(blocks[1].seq_hash), &blocks[2..]);
    assert_eq!(store.admit(a, 0, batch(5), &tail), BatchOutcome::Applied);
    assert_eq!(
        store.lane_counts(a, 0).expect("a lane").parentless_stores,
        1
    );
    assert_eq!(held(&store, MODEL, &[1, 2, 3, 4]), BTreeMap::new());
    assert_eq!(held(&store, MODEL, &[3, 4]), BTreeMap::from([(a, 2)]));
    assert_eq!(store.indexed_blocks(a, 0), 2);
}

#[test]
fn a_row_past_the_ceiling_stands_counted_until_its_own_write_finds_a_place() {
    let store = MemoryStore::new();
    let blocks = chain(&[1]);
    let ids: Vec<WorkerId> = (0..INDEX_WORKERS)
        .map(|n| admitted(&store, &format!("w{n}")))
        .collect();
    assert_eq!(store.unindexed_rows(), 0);
    let over = admitted(&store, "over");
    assert_eq!(store.unindexed_rows(), 1);
    assert_eq!(
        store.admit(over, 0, batch(1), &stored(None, &blocks)),
        BatchOutcome::Skipped
    );
    // Written again while the index is full: one row stands out, however
    // often it is written.
    let rewritten = |priority| {
        store
            .upsert(WorkerFacts {
                priority,
                ..publisher("over", &[MODEL])
            })
            .id()
    };
    for priority in [1, 2] {
        assert_eq!(rewritten(priority), over);
        assert_eq!(store.unindexed_rows(), 1);
    }
    // A row leaves, and no write names the row past the ceiling: it is
    // still out, and the place is the next written row's.
    let leave = |id| {
        let revision = WorkersStore::get(&store, id).expect("the row").revision;
        store.remove(id, revision).expect("removed");
    };
    leave(ids[3]);
    assert_eq!(store.unindexed_rows(), 1);
    assert_eq!(
        store.admit(over, 0, batch(1), &stored(None, &blocks)),
        BatchOutcome::Skipped
    );
    let next = admitted(&store, "next");
    assert_eq!(
        store.admit(next, 0, batch(1), &stored(None, &blocks)),
        BatchOutcome::Applied
    );
    assert_eq!(rewritten(3), over);
    assert_eq!(store.unindexed_rows(), 1);
    // Another leaves: the row's own next write finds the place.
    leave(ids[4]);
    assert_eq!(rewritten(4), over);
    assert_eq!(store.unindexed_rows(), 0);
    assert_eq!(
        store.admit(over, 0, batch(1), &stored(None, &blocks)),
        BatchOutcome::Applied
    );
    assert_eq!(
        held(&store, MODEL, &[1]),
        BTreeMap::from([(next, 1), (over, 1)])
    );
}

/// A stand-in for an index at a size cap: it places `room` blocks more and
/// declines the rest.
struct Capped {
    room: usize,
}

impl Blocks for Capped {
    fn stored(
        &mut self,
        blocks: &[StoredBlock],
        _parent: Option<SequenceHash>,
    ) -> Result<(), ApplyError> {
        let placed = blocks.len().min(self.room);
        self.room -= placed;
        if placed < blocks.len() {
            return Err(ApplyError::ArenaCapReached { placed });
        }
        Ok(())
    }

    fn removed(&mut self, _hashes: &[SequenceHash]) {}

    fn cleared(&mut self) {}
}

#[test]
fn blocks_an_index_declines_at_a_size_cap_are_counted() {
    let mut cursor = RankCursor::default();
    let mut capped = Capped { room: 3 };
    let mut counts = LaneCounts::default();
    // Five blocks where three fit: two are declined.
    let (outcome, _) = admit(
        &mut cursor,
        &mut capped,
        batch(1),
        &stored(None, &chain(&[1, 2, 3, 4, 5])),
        &mut counts,
    );
    assert_eq!(outcome, BatchOutcome::Applied);
    assert_eq!(counts.refused_blocks, 2);
    // No room is left: every block of the next store is declined.
    let (outcome, _) = admit(
        &mut cursor,
        &mut capped,
        batch(2),
        &stored(None, &chain(&[6, 7, 8, 9])),
        &mut counts,
    );
    assert_eq!(outcome, BatchOutcome::Applied);
    assert_eq!(
        counts,
        LaneCounts {
            parentless_stores: 0,
            refused_blocks: 6,
        }
    );
}

#[test]
fn a_snapshots_start_a_long_gap_and_a_clear_among_the_events_each_empty_the_rank() {
    let store = MemoryStore::new();
    let a = admitted(&store, "a");
    let old = chain(&[1, 2, 3]);
    let new = chain(&[7, 8]);
    store.admit(a, 0, batch(1), &stored(None, &old));
    // A snapshot's start empties the rank, and each of its chunks adds.
    let chunk = BatchFacts {
        sequence: 40,
        clears: false,
        snapshot: Some(SnapshotChunk {
            unknown_before: false,
        }),
    };
    store.resync(a, None, ResyncReason::Snapshot);
    assert_eq!(held(&store, MODEL, &[1, 2, 3]), BTreeMap::new());
    assert_eq!(
        store.admit(a, 0, chunk, &stored(None, &new[..1])),
        BatchOutcome::Applied
    );
    let rest = stored(Some(new[0].seq_hash), &new[1..]);
    assert_eq!(store.admit(a, 0, chunk, &rest), BatchOutcome::Applied);
    assert_eq!(held(&store, MODEL, &[7, 8]), BTreeMap::from([(a, 2)]));
    assert_eq!(store.resyncs(ResyncReason::Snapshot), 1);
    // A gap no replay fills, longer than the state is kept across.
    assert_eq!(
        store.admit(a, 0, batch(50), &stored(None, &old)),
        BatchOutcome::Gap {
            expected: 41,
            received: 50
        }
    );
    let far = 42 + GAP_CLEAR_THRESHOLD;
    assert_eq!(
        store.admit(a, 0, batch(far), &stored(None, &old[..1])),
        BatchOutcome::Applied
    );
    assert_eq!(held(&store, MODEL, &[7, 8]), BTreeMap::new());
    assert_eq!(held(&store, MODEL, &[1]), BTreeMap::from([(a, 1)]));
    assert_eq!(store.resyncs(ResyncReason::GapCleared), 1);
    // A clear among the events of a batch below the cursor is a restarted
    // publisher, whatever the header says: the blocks stored after it stay.
    let restarted = vec![
        KvEvent::Cleared,
        KvEvent::Stored {
            parent: None,
            blocks: new.clone(),
        },
    ];
    assert_eq!(
        store.admit(a, 0, batch(far - 5), &restarted),
        BatchOutcome::Applied
    );
    assert_eq!(held(&store, MODEL, &[1]), BTreeMap::new());
    assert_eq!(held(&store, MODEL, &[7, 8]), BTreeMap::from([(a, 2)]));
    assert_eq!(store.resyncs(ResyncReason::PublisherRestart), 1);
    assert_eq!(store.cursor(a, 0).expect("heard").applied, far - 5);
}

/// One worker of the reference index, written by the lane's own admission.
struct Reference<'a> {
    index: &'a mut ReferenceIndexer,
    worker: u32,
}

impl Blocks for Reference<'_> {
    fn stored(
        &mut self,
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
    ) -> Result<(), ApplyError> {
        self.index.apply_stored(self.worker, blocks, parent)
    }

    fn removed(&mut self, hashes: &[SequenceHash]) {
        self.index.apply_removed(self.worker, hashes);
    }

    fn cleared(&mut self) {
        self.index.apply_cleared(self.worker);
    }
}

/// SplitMix64: a stream's only randomness, so a failing seed replays.
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const WORKERS: usize = 5;
const PROMPTS: usize = 8;
const LENGTH: usize = 12;
const STEPS: usize = 1_500;

/// The prompts the engines serve: each starts from nothing or from the
/// head of an earlier one, so the workers' chains share prefixes and fork.
fn prompts(mix: &mut Mix) -> Vec<Vec<u64>> {
    let mut all: Vec<Vec<u64>> = Vec::with_capacity(PROMPTS);
    for _ in 0..PROMPTS {
        let mut prompt = if all.is_empty() || mix.below(3) == 0 {
            Vec::new()
        } else {
            let from = &all[mix.below(all.len())];
            from[..=mix.below(LENGTH - 1)].to_vec()
        };
        while prompt.len() < LENGTH {
            prompt.push(mix.next() | 1);
        }
        all.push(prompt);
    }
    all
}

/// One engine's cache: the blocks it holds by name, each with its parent's
/// name, and its publisher's counter.
#[derive(Default)]
struct Engine {
    held: BTreeMap<SequenceHash, (Option<SequenceHash>, ContentHash, usize)>,
    sequence: u64,
}

impl Engine {
    /// Serve the first `upto` blocks of `prompt`: every block not held is
    /// stored, each run of them under the block before it.
    fn serve(&mut self, prompt: &[u64], upto: usize) -> Vec<KvEvent> {
        let blocks = chain(&prompt[..upto]);
        let mut events = Vec::new();
        let mut run: Vec<StoredBlock> = Vec::new();
        let mut run_parent = None;
        for (depth, block) in blocks.iter().enumerate() {
            let parent = depth.checked_sub(1).map(|at| blocks[at].seq_hash);
            if self.held.contains_key(&block.seq_hash) {
                if !run.is_empty() {
                    events.push(KvEvent::Stored {
                        parent: run_parent,
                        blocks: std::mem::take(&mut run),
                    });
                }
                continue;
            }
            if run.is_empty() {
                run_parent = parent;
            }
            self.held
                .insert(block.seq_hash, (parent, block.content_hash, depth));
            run.push(*block);
        }
        if !run.is_empty() {
            events.push(KvEvent::Stored {
                parent: run_parent,
                blocks: run,
            });
        }
        events
    }

    /// Evict up to `count` blocks, wherever they sit in their chains.
    fn evict(&mut self, mix: &mut Mix, count: usize) -> Vec<KvEvent> {
        let mut names = Vec::new();
        for _ in 0..count.min(self.held.len()) {
            let at = mix.below(self.held.len());
            if let Some(&name) = self.held.keys().nth(at) {
                self.held.remove(&name);
                names.push(name);
            }
        }
        if names.is_empty() {
            Vec::new()
        } else {
            vec![KvEvent::Removed(names)]
        }
    }

    /// The cache as a relay retells it: every block under its parent,
    /// parents first, in two chunks.
    fn snapshot(&self) -> [Vec<KvEvent>; 2] {
        let mut blocks: Vec<_> = self.held.iter().collect();
        blocks.sort_by_key(|(name, (_, _, depth))| (*depth, **name));
        let half = blocks.len().div_ceil(2);
        let mut chunks = [Vec::new(), Vec::new()];
        for (n, (name, (parent, content, _))) in blocks.into_iter().enumerate() {
            chunks[usize::from(n >= half)].push(KvEvent::Stored {
                parent: *parent,
                blocks: vec![StoredBlock {
                    seq_hash: *name,
                    content_hash: *content,
                }],
            });
        }
        chunks
    }
}

/// What the wire does with a worker's next batches.
enum Wire {
    Live,
    /// A batch held back: the next reveals the gap and the replay fills it.
    Held(BatchFacts, Vec<KvEvent>),
    /// A batch lost for good: the next reveals the gap and is lost with it.
    Lost,
    /// The gap was revealed and no replay comes: the next is lost history.
    Revealed,
}

/// The store's index answers every lookup as the reference index does when
/// both are written by the lane's admission, batch for batch: the stream
/// evicts in the middle of chains, clears, restarts its publisher, repeats,
/// holds back and loses batches, and is retold by snapshots.
#[test]
fn the_sinks_apply_path_holds_the_index_to_the_reference_over_lossy_streams() {
    for seed in 1..=8u64 {
        let mut mix = Mix(seed);
        let prompts = prompts(&mut mix);
        let store = MemoryStore::new();
        let ids: Vec<WorkerId> = (0..WORKERS)
            .map(|n| admitted(&store, &format!("w{n}")))
            .collect();
        let mut reference = ReferenceIndexer::new();
        let mut cursors = [RankCursor::default(); WORKERS];
        let mut counts = [LaneCounts::default(); WORKERS];
        let mut engines: Vec<Engine> = (0..WORKERS).map(|_| Engine::default()).collect();
        let mut wires: Vec<Wire> = (0..WORKERS).map(|_| Wire::Live).collect();
        let mut compared = 0u64;
        for step in 0..STEPS {
            let w = mix.below(WORKERS);
            let mut deliver = |facts: BatchFacts, events: &[KvEvent]| {
                let outcome = store.admit(ids[w], 0, facts, events);
                let mut blocks = Reference {
                    index: &mut reference,
                    worker: w as u32,
                };
                let (expected, _) =
                    admit(&mut cursors[w], &mut blocks, facts, events, &mut counts[w]);
                assert_eq!(outcome, expected, "seed {seed} step {step}");
            };
            let engine = &mut engines[w];
            let events = match mix.below(12) {
                0 | 1 => engine.evict(&mut mix, 1 + step % 4),
                2 if mix.below(6) == 0 => {
                    engine.held.clear();
                    vec![KvEvent::Cleared]
                }
                _ => {
                    let prompt = &prompts[mix.below(PROMPTS)];
                    engine.serve(prompt, 1 + mix.below(LENGTH))
                }
            };
            if events.is_empty() {
                continue;
            }
            engine.sequence += 1;
            let facts = batch(engine.sequence);
            match std::mem::replace(&mut wires[w], Wire::Live) {
                Wire::Held(held_facts, held_events) => {
                    deliver(facts, &events);
                    deliver(held_facts, &held_events);
                    deliver(facts, &events);
                }
                Wire::Lost => {
                    deliver(facts, &events);
                    wires[w] = Wire::Revealed;
                }
                Wire::Revealed => deliver(facts, &events),
                Wire::Live => match mix.below(40) {
                    0 => {
                        deliver(facts, &events);
                        deliver(facts, &events);
                    }
                    1 => wires[w] = Wire::Held(facts, events),
                    2 => wires[w] = Wire::Lost,
                    3 => {
                        // A missed stretch too long to keep the state.
                        engine.sequence += 2_000;
                        wires[w] = Wire::Lost;
                    }
                    4 => {
                        // The publisher starts over on a new connection,
                        // its cache empty.
                        deliver(facts, &events);
                        engine.held.clear();
                        engine.sequence = 0;
                        store.reconnected(ids[w]);
                        cursors[w].reconnected();
                    }
                    5 => {
                        // A relay retells the cache: the start is said to
                        // the row, then each chunk adds.
                        store.resync(ids[w], None, ResyncReason::Snapshot);
                        cursors[w].reset();
                        let mut blocks = Reference {
                            index: &mut reference,
                            worker: w as u32,
                        };
                        blocks.cleared();
                        let facts = BatchFacts {
                            snapshot: Some(SnapshotChunk {
                                unknown_before: false,
                            }),
                            ..facts
                        };
                        for chunk in &engine.snapshot() {
                            let outcome = store.admit(ids[w], 0, facts, chunk);
                            let (expected, _) =
                                admit(&mut cursors[w], &mut blocks, facts, chunk, &mut counts[w]);
                            assert_eq!(outcome, expected, "seed {seed} step {step}");
                        }
                        // The engine's own cache is the answer the reference
                        // cannot give: a retelling leaves all of it, alone.
                        assert_eq!(
                            store.indexed_blocks(ids[w], 0),
                            engine.held.len(),
                            "seed {seed} step {step}: the retold cache"
                        );
                    }
                    _ => deliver(facts, &events),
                },
            }
            for _ in 0..3 {
                let prompt = &prompts[mix.below(PROMPTS)];
                let mut request = prompt[..=mix.below(LENGTH)].to_vec();
                if mix.below(5) == 0 {
                    // A request that leaves the prompt at its last block.
                    if let Some(last) = request.last_mut() {
                        *last ^= 2;
                    }
                }
                let hashes: Vec<ContentHash> = request.iter().map(|&c| ContentHash(c)).collect();
                let expected: BTreeMap<WorkerId, u32> = reference
                    .find_matches(&hashes)
                    .into_iter()
                    .map(|(worker, blocks)| (ids[worker as usize], blocks))
                    .collect();
                assert_eq!(
                    held(&store, MODEL, &request),
                    expected,
                    "seed {seed} step {step}: the request {request:?}"
                );
                compared += u64::from(!expected.is_empty());
            }
            for (worker, id) in ids.iter().enumerate() {
                assert_eq!(
                    store.indexed_blocks(*id, 0),
                    reference.worker_block_count(worker as u32),
                    "seed {seed} step {step}: the blocks of worker {worker}"
                );
                assert_eq!(store.lane_counts(*id, 0), Some(counts[worker]));
            }
        }
        assert!(
            compared > 500,
            "seed {seed}: {compared} lookups found a holder"
        );
        assert!(
            counts.iter().any(|count| count.parentless_stores > 0),
            "seed {seed}: no store met a parent the index had lost"
        );
    }
}

/// Rows apply at once, each through its own lane, while a reader looks the
/// index up: what each holds at the end is what it stored.
#[test]
fn rows_apply_at_once_beside_a_reader() {
    const ROWS: usize = 4;
    const BATCHES: u64 = 400;
    let store = MemoryStore::new();
    let ids: Vec<WorkerId> = (0..ROWS)
        .map(|n| admitted(&store, &format!("w{n}")))
        .collect();
    // One shared head of four blocks, then each row's own tail.
    let head = [11, 12, 13, 14];
    std::thread::scope(|scope| {
        for (row, &id) in ids.iter().enumerate() {
            let store = &store;
            scope.spawn(move || {
                let mut contents = head.to_vec();
                let mut blocks = chain(&contents);
                store.admit(id, 0, batch(1), &stored(None, &blocks));
                for sequence in 2..=BATCHES {
                    contents.push(1_000 * (row as u64 + 1) + sequence);
                    let next = chain(&contents);
                    let parent = blocks.last().map(|block| block.seq_hash);
                    let outcome = store.admit(
                        id,
                        0,
                        batch(sequence),
                        &stored(parent, &next[blocks.len()..]),
                    );
                    assert_eq!(outcome, BatchOutcome::Applied);
                    blocks = next;
                }
            });
        }
        let store = &store;
        scope.spawn(move || {
            for _ in 0..2_000 {
                for (_, blocks) in store.kv().overlaps(MODEL, &head) {
                    assert!(blocks <= 4);
                }
            }
        });
    });
    assert_eq!(
        held(&store, MODEL, &head),
        ids.iter().map(|&id| (id, 4)).collect()
    );
    for (row, &id) in ids.iter().enumerate() {
        assert_eq!(store.indexed_blocks(id, 0), 3 + BATCHES as usize);
        let mut own = head.to_vec();
        own.extend((2..=BATCHES).map(|sequence| 1_000 * (row as u64 + 1) + sequence));
        let all = held(&store, MODEL, &own);
        assert_eq!(all[&id], own.len() as u32);
        assert!(all
            .iter()
            .all(|(other, blocks)| *other == id || *blocks == 4));
    }
}

#[test]
fn a_worker_that_holds_blocks_stores_under_a_parent_it_does_not_hold() {
    let store = MemoryStore::new();
    let a = admitted(&store, "a");
    let first = chain(&[1, 2]);
    assert_eq!(
        store.admit(a, 0, batch(1), &stored(None, &first)),
        BatchOutcome::Applied
    );
    // The worker holds blocks; the next store names a parent it never heard.
    let other = chain(&[7, 8, 9, 10]);
    let tail = stored(Some(other[1].seq_hash), &other[2..]);
    assert_eq!(store.admit(a, 0, batch(2), &tail), BatchOutcome::Applied);
    assert_eq!(
        store.lane_counts(a, 0).expect("a lane").parentless_stores,
        1
    );
    assert_eq!(store.indexed_blocks(a, 0), 4);
    assert_eq!(held(&store, MODEL, &[9, 10]), BTreeMap::from([(a, 2)]));
    assert_eq!(held(&store, MODEL, &[1, 2]), BTreeMap::from([(a, 2)]));
}

#[test]
fn a_row_that_stops_asking_for_events_leaves_its_lane_and_its_blocks() {
    let store = MemoryStore::new();
    let a = admitted(&store, "a");
    let blocks = chain(&[1, 2, 3]);
    assert_eq!(
        store.admit(a, 0, batch(1), &stored(None, &blocks)),
        BatchOutcome::Applied
    );
    assert_eq!(
        (store.kv().indexes(), store.kv().held_blocks(MODEL)),
        (1, 3)
    );
    let again = store
        .upsert(WorkerFacts {
            signals: Signals::default(),
            ..publisher("a", &[MODEL])
        })
        .id();
    assert_eq!(again, a, "the same row");
    assert_eq!(store.indexed_blocks(a, 0), 0);
    assert_eq!(
        (store.kv().indexes(), store.kv().held_blocks(MODEL)),
        (0, 0)
    );
    assert_eq!(
        store.admit(a, 0, batch(2), &stored(None, &blocks)),
        BatchOutcome::Skipped
    );
}

#[test]
fn a_row_moved_to_another_first_model_leaves_nothing_in_the_old_index() {
    let store = MemoryStore::new();
    let two = store.upsert(publisher("two", &["first", "second"])).id();
    let blocks = chain(&[1, 2]);
    store.admit(two, 0, batch(1), &stored(None, &blocks));
    assert_eq!(
        (store.kv().indexes(), store.kv().held_blocks("first")),
        (1, 2)
    );
    assert_eq!(
        store.upsert(publisher("two", &["second", "first"])).id(),
        two
    );
    // The old index went with its last lane; the new one starts empty.
    assert_eq!(
        (store.kv().indexes(), store.kv().held_blocks("first")),
        (1, 0)
    );
    assert_eq!(store.kv().held_blocks("second"), 0);
}

/// A snapshot is an engine's and a lane is a rank's. The collector says the
/// start to every row of the engine, and the relay's chunks, numbered over
/// the ranks with the clear in the first alone, only add. Each rank holds
/// what the snapshot holds for it, and a rank it holds nothing for is empty
/// with its cursor started over.
#[test]
fn a_snapshot_said_to_every_rank_of_its_engine_leaves_each_what_the_snapshot_holds() {
    let store = MemoryStore::new();
    let ranks: Vec<WorkerId> = (0..3)
        .map(|rank| store.upsert(ranked("e", rank, 3)).id())
        .collect();
    let old = chain(&[1, 2, 3]);
    for (rank, &row) in (0u16..).zip(&ranks) {
        assert_eq!(
            store.admit(row, rank, batch(5), &stored(None, &old)),
            BatchOutcome::Applied
        );
    }
    for &row in &ranks {
        store.resync(row, None, ResyncReason::Snapshot);
    }
    // The relay's first chunk is rank 0's and carries the clear; its second
    // is rank 1's; rank 2 holds nothing now, so no chunk is its own.
    let chunk = BatchFacts {
        sequence: 40,
        clears: false,
        snapshot: Some(SnapshotChunk {
            unknown_before: false,
        }),
    };
    let new = chain(&[7, 8]);
    let first = vec![
        KvEvent::Cleared,
        KvEvent::Stored {
            parent: None,
            blocks: new.clone(),
        },
    ];
    assert_eq!(
        store.admit(ranks[0], 0, chunk, &first),
        BatchOutcome::Applied
    );
    assert_eq!(
        store.admit(ranks[1], 1, chunk, &stored(None, &new)),
        BatchOutcome::Applied
    );
    let blocks: Vec<usize> = (0u16..)
        .zip(&ranks)
        .map(|(rank, &row)| store.indexed_blocks(row, rank))
        .collect();
    assert_eq!(blocks, vec![2, 2, 0]);
    assert_eq!(store.resyncs(ResyncReason::Snapshot), 3, "one a row");
    // The rank the snapshot left out starts over: a batch below where it
    // stood is its first, not a late one.
    assert_eq!(
        store.admit(ranks[2], 2, batch(3), &stored(None, &old[..1])),
        BatchOutcome::Applied
    );
    assert_eq!(store.indexed_blocks(ranks[2], 2), 1);
}

/// A second snapshot with no live batch between: its start is said again,
/// and nothing of the first is left. A chunk with no start said empties
/// nothing: it adds to what its rank holds.
#[test]
fn a_second_snapshot_leaves_nothing_of_the_first_and_a_chunk_alone_only_adds() {
    let store = MemoryStore::new();
    let a = admitted(&store, "a");
    let chunk = |sequence| BatchFacts {
        sequence,
        clears: false,
        snapshot: Some(SnapshotChunk {
            unknown_before: false,
        }),
    };
    let (first, second) = (chain(&[1, 2, 3]), chain(&[7, 8]));
    store.resync(a, None, ResyncReason::Snapshot);
    assert_eq!(
        store.admit(a, 0, chunk(40), &stored(None, &first)),
        BatchOutcome::Applied
    );
    store.resync(a, None, ResyncReason::Snapshot);
    assert_eq!(
        store.admit(a, 0, chunk(40), &stored(None, &second)),
        BatchOutcome::Applied
    );
    assert_eq!(held(&store, MODEL, &[1, 2, 3]), BTreeMap::new());
    assert_eq!(held(&store, MODEL, &[7, 8]), BTreeMap::from([(a, 2)]));
    assert_eq!(
        store.admit(a, 0, chunk(41), &stored(None, &first)),
        BatchOutcome::Applied
    );
    assert_eq!(store.indexed_blocks(a, 0), 5);
    assert_eq!(store.resyncs(ResyncReason::Snapshot), 2);
    assert_eq!(store.cursor(a, 0).expect("heard").applied, 41);
}

/// What a batch that clears leaves owes nothing to what came before it,
/// wherever the clear stands among its events: below the cursor such a batch
/// is a restarted publisher's, applied whole, and the stream goes on from
/// it.
#[test]
fn a_clear_in_the_middle_of_a_batch_below_the_cursor_is_a_restart() {
    let store = MemoryStore::new();
    let a = admitted(&store, "a");
    let (old, new) = (chain(&[1, 2, 3]), chain(&[7, 8]));
    store.admit(a, 0, batch(9), &stored(None, &old));
    let middle = vec![
        KvEvent::Stored {
            parent: None,
            blocks: chain(&[5]),
        },
        KvEvent::Cleared,
        KvEvent::Stored {
            parent: None,
            blocks: new,
        },
    ];
    assert_eq!(store.admit(a, 0, batch(4), &middle), BatchOutcome::Applied);
    assert_eq!(held(&store, MODEL, &[1, 2, 3]), BTreeMap::new());
    assert_eq!(held(&store, MODEL, &[7, 8]), BTreeMap::from([(a, 2)]));
    assert_eq!(store.indexed_blocks(a, 0), 2);
    assert_eq!(store.resyncs(ResyncReason::PublisherRestart), 1);
    assert_eq!(store.cursor(a, 0).expect("heard").applied, 4);
    assert_eq!(
        store.admit(a, 0, batch(5), &stored(None, &old[..1])),
        BatchOutcome::Applied
    );
}

/// A row written at another rank starts over there: its lane is made anew
/// and empty, and a batch of its old rank is skipped.
#[test]
fn a_row_written_at_another_rank_starts_over_there() {
    let store = MemoryStore::new();
    let row = store.upsert(ranked("e", 0, 2)).id();
    let blocks = chain(&[1, 2, 3]);
    assert_eq!(
        store.admit(row, 0, batch(1), &stored(None, &blocks)),
        BatchOutcome::Applied
    );
    let moved = WorkerFacts {
        dp_rank: 1,
        ..ranked("e", 0, 2)
    };
    assert_eq!(store.upsert(moved).id(), row, "the same row");
    assert_eq!(held(&store, MODEL, &[1, 2, 3]), BTreeMap::new());
    assert_eq!(
        store.admit(row, 0, batch(2), &stored(None, &blocks)),
        BatchOutcome::Skipped,
        "its old rank"
    );
    assert_eq!(store.cursor(row, 1), None, "unheard at its new rank");
    assert_eq!(
        store.admit(row, 1, batch(1), &stored(None, &blocks[..1])),
        BatchOutcome::Applied
    );
    assert_eq!(store.indexed_blocks(row, 1), 1);
}

/// The rows a decision is handed in the affinity column's place.
mod named {
    use std::{
        cell::Cell,
        rc::Rc,
        sync::{atomic::Ordering, Arc},
        time::{Duration, Instant},
    };

    use super::{admitted, batch, chain, publisher, stored, MODEL};
    use crate::{
        store::{
            index::{seam, PoolIndex},
            Ask, BatchOutcome, GatewayId, KvEventSink, LoadSink, LoadSource, MemoryStore, Named,
            Naming, Reachability, Signals, ViewBuffers, WorkerFacts, WorkerRow, WorkersStore,
        },
        view::{Candidate, LoadRow, Role, WireModes, WorkerId},
        AppliedSpec, PolicySpec,
    };

    /// Another gateway than the one most cases decide for.
    const OTHER: GatewayId = GatewayId(9);

    /// The preset whose picker takes a sample of sixteen, as a generation:
    /// its warm-up stage rules who warms (a level of 1,024 blocks, half of
    /// it thin, sixty seconds young).
    fn sampled(generation: u64) -> AppliedSpec {
        let mut document = PolicySpec::preset("cache_aware").expect("a preset of this build");
        document.generation = generation;
        AppliedSpec::validate(&document).expect("a preset validates")
    }

    /// `facts` registered: the row's id.
    fn registered(store: &MemoryStore, facts: WorkerFacts) -> WorkerId {
        let id = store.upsert(facts).id();
        // The build this registration owes, asked: the cases read the row
        // at once.
        store.publish_now();
        id
    }

    /// What one decision is handed, by worker: its rows, the holders named
    /// with their depths, the warming rows named, and the bit.
    struct Handed {
        rows: Vec<WorkerId>,
        holders: Vec<(WorkerId, u32)>,
        warming: Vec<WorkerId>,
        thin: bool,
    }

    /// The naming of `request` for a decision of `gateway` that states
    /// `wires`, under a cap of `cap` rows, made now.
    fn handed(
        store: &MemoryStore,
        gateway: GatewayId,
        wires: Option<WireModes>,
        request: &[u64],
        cut: (u32, u64),
    ) -> Handed {
        handed_at(store, gateway, wires, request, cut, Instant::now())
    }

    /// The naming of `request` for a decision of `gateway` that states
    /// `wires`, under a cap of `cap` rows, made at `at`: the view built,
    /// narrowed as the decider narrows it, then named.
    fn handed_at(
        store: &MemoryStore,
        gateway: GatewayId,
        wires: Option<WireModes>,
        request: &[u64],
        (cap, seed): (u32, u64),
        at: Instant,
    ) -> Handed {
        let mut view = ViewBuffers::default();
        let mut named = Named::default();
        store.deciding(|world| {
            let ask = Ask {
                gateway,
                now: at,
                pinned: None,
                affinity: false,
            };
            let built = world
                .build(MODEL, Role::Single, &ask, &mut view)
                .expect("the pool");
            if let Some(wires) = wires {
                view.candidates
                    .retain(|row| row.wire_modes.intersects(wires));
            }
            let naming = Naming {
                gateway,
                block_hashes: request,
                seed,
                cap,
                wires,
                now_micros: built.now_micros,
            };
            world.name(MODEL, Role::Single, &naming, &view.candidates, &mut named);
        });
        let rows: Vec<WorkerId> = view.candidates.iter().map(|row| row.id).collect();
        let places = |list: &[u32]| list.windows(2).all(|pair| pair[0] < pair[1]);
        let holders: Vec<u32> = named.holders.iter().map(|holder| holder.row).collect();
        assert!(
            places(&holders) && places(&named.warming),
            "ascending, once"
        );
        Handed {
            holders: named
                .holders
                .iter()
                .map(|holder| (rows[holder.row as usize], holder.overlap_blocks))
                .collect(),
            warming: named
                .warming
                .iter()
                .map(|&row| rows[row as usize])
                .collect(),
            thin: named.thin,
            rows,
        }
    }

    /// `id` holds `blocks`, by one batch.
    fn holds(store: &MemoryStore, id: WorkerId, contents: &[u64]) {
        let outcome = store.admit(id, 0, batch(1), &stored(None, &chain(contents)));
        assert_eq!(outcome, BatchOutcome::Applied);
    }

    /// A decision is handed its request's deepest holders by their place in
    /// its rows: whole depths from the deepest while they fit, and of the
    /// depth the cap cuts as many as fit.
    #[test]
    fn a_decision_is_handed_its_requests_deepest_holders_by_their_place() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        let ids: Vec<WorkerId> = (0..6).map(|n| admitted(&store, &format!("w{n}"))).collect();
        // Two hold the whole request, three its first two blocks, one none.
        for (n, &id) in ids.iter().enumerate().take(5) {
            holds(&store, id, if n < 2 { &[1, 2, 3, 4] } else { &[1, 2] });
        }
        let request = [1, 2, 3, 4];
        let mut every: Vec<(WorkerId, u32)> = (0..5)
            .map(|n| (ids[n], if n < 2 { 4 } else { 2 }))
            .collect();
        every.sort_unstable();
        let whole = handed(&store, GatewayId::LOCAL, None, &request, (16, 7));
        assert_eq!(whole.rows.len(), 6);
        assert_eq!(whole.holders, every);
        // No cap holds every holder.
        let uncapped = handed(&store, GatewayId::LOCAL, None, &request, (0, 7));
        assert_eq!(uncapped.holders, every);
        // Under a cap of three: the two deepest and one of the three under
        // them, whichever the seed draws.
        let mut drawn = std::collections::BTreeSet::new();
        for seed in 0..40 {
            let cut = handed(&store, GatewayId::LOCAL, None, &request, (3, seed)).holders;
            let deep = cut.iter().filter(|holder| holder.1 == 4).count();
            assert_eq!((cut.len(), deep), (3, 2), "seed {seed}");
            assert!(cut.iter().all(|holder| every.contains(holder)));
            drawn.extend(
                cut.iter()
                    .filter(|holder| holder.1 == 2)
                    .map(|holder| holder.0),
            );
        }
        assert_eq!(drawn.len(), 3, "each of the three is drawn by some seed");
        // A request nobody holds, and a request with no block: nobody named.
        assert!(handed(&store, GatewayId::LOCAL, None, &[9], (16, 7))
            .holders
            .is_empty());
        assert!(handed(&store, GatewayId::LOCAL, None, &[], (16, 7))
            .holders
            .is_empty());
    }

    /// A row the decision cannot reach is not named, and the cap is filled
    /// from the rows it can: a gateway that states wires reads the holders
    /// that share one with it, and no gateway reads another's host-local
    /// row.
    #[test]
    fn the_holders_are_named_among_the_rows_a_decision_reaches() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        let member = |key: &str, wires: WireModes, reach: Reachability| {
            let facts = WorkerFacts {
                wire_modes: wires,
                reachable_from: reach,
                ..publisher(key, &[MODEL])
            };
            let id = registered(&store, facts);
            holds(&store, id, &[1, 2, 3]);
            id
        };
        let http = member("http", WireModes::HTTP, Reachability::Any);
        let grpc = member("grpc", WireModes::GRPC, Reachability::Any);
        let wires = WireModes::HTTP.union(WireModes::GRPC);
        let both = member("both", wires, Reachability::Any);
        let local = member("local", WireModes::GRPC, Reachability::Gateway(OTHER));
        let named = |gateway, wires, cap, seed| {
            let got = handed(&store, gateway, wires, &[1, 2, 3], (cap, seed));
            got.holders
                .iter()
                .map(|holder| holder.0)
                .collect::<Vec<_>>()
        };
        let sorted = |mut ids: Vec<WorkerId>| {
            ids.sort_unstable();
            ids
        };
        // Each gateway's own rows, whole.
        let own = sorted(vec![http, grpc, both]);
        assert_eq!(named(GatewayId::LOCAL, None, 16, 1), own);
        let theirs = sorted(vec![http, grpc, both, local]);
        assert_eq!(named(OTHER, None, 16, 1), theirs);
        // Under a cap of two a gateway that speaks gRPC alone is handed the
        // two holders it dials, whatever the seed: the row that speaks HTTP
        // alone and the other gateway's row take no place of the cap.
        let dialed = sorted(vec![grpc, both]);
        for seed in 0..40 {
            let cut = named(GatewayId::LOCAL, Some(WireModes::GRPC), 2, seed);
            assert_eq!(cut, dialed, "seed {seed}");
            let one = named(GatewayId::LOCAL, None, 1, seed);
            assert!(one.len() == 1 && one[0] != local, "seed {seed}: {one:?}");
        }
        // A pool that holds no row foreign to its gateway is narrowed by that
        // gateway's wires all the same: the row that speaks HTTP alone is not
        // read.
        let reached = sorted(vec![grpc, both, local]);
        assert_eq!(named(OTHER, Some(WireModes::GRPC), 16, 1), reached);
    }

    /// What a decision that holds the world of before is named for
    /// `request` under a cap and each seed of `seeds` once `then` has run
    /// under it: the holders by their workers, a seed after another, and
    /// the workers of that world's rows.
    fn named_under_a_kept_world_each(
        store: &MemoryStore,
        request: &[u64],
        cap: u32,
        seeds: &[u64],
        then: impl FnOnce(),
    ) -> (Vec<Vec<(WorkerId, u32)>>, Vec<WorkerId>) {
        let mut view = ViewBuffers::default();
        let mut named = Named::default();
        let mut each = Vec::with_capacity(seeds.len());
        let before: Vec<WorkerId> = store.deciding(|world| {
            let ask = Ask {
                gateway: GatewayId::LOCAL,
                now: Instant::now(),
                pinned: None,
                affinity: false,
            };
            let built = world
                .build(MODEL, Role::Single, &ask, &mut view)
                .expect("the pool");
            then();
            let before: Vec<WorkerId> = view.candidates.iter().map(|row| row.id).collect();
            for &seed in seeds {
                let naming = Naming {
                    gateway: GatewayId::LOCAL,
                    block_hashes: request,
                    seed,
                    cap,
                    wires: None,
                    now_micros: built.now_micros,
                };
                world.name(MODEL, Role::Single, &naming, &view.candidates, &mut named);
                let holders = named.holders.iter();
                each.push(
                    holders
                        .map(|holder| (before[holder.row as usize], holder.overlap_blocks))
                        .collect(),
                );
            }
            before
        });
        (each, before)
    }

    /// The same under one seed.
    fn named_under_a_kept_world(
        store: &MemoryStore,
        request: &[u64],
        (cap, seed): (u32, u64),
        then: impl FnOnce(),
    ) -> (Vec<(WorkerId, u32)>, Vec<WorkerId>) {
        let (mut each, before) = named_under_a_kept_world_each(store, request, cap, &[seed], then);
        (each.pop().expect("one seed"), before)
    }

    /// A row's id in the model's index, as the world of now holds it.
    fn index_id(store: &MemoryStore, id: WorkerId) -> u32 {
        store.deciding(|world| {
            let rows = handed(store, GatewayId::LOCAL, None, &[], (16, 7)).rows;
            let row = rows.iter().position(|&row| row == id).expect("a row");
            let standing = world.standing(MODEL, Role::Single).expect("a standing");
            standing.tables.id_of[row]
        })
    }

    /// What that decision is named under each seed of `seeds` once `left`
    /// has left under its world, a row has taken its index id, and `then`
    /// has said what the taker and the others hold: the holders by their
    /// workers, a seed after another, and the taker.
    fn named_past_an_id_taken_again_each(
        store: &MemoryStore,
        left: WorkerId,
        request: &[u64],
        cap: u32,
        seeds: &[u64],
        then: impl FnOnce(WorkerId),
    ) -> (Vec<Vec<(WorkerId, u32)>>, WorkerId) {
        let mut taker = None;
        let (each, before) = named_under_a_kept_world_each(store, request, cap, seeds, || {
            let freed = index_id(store, left);
            let revision = WorkersStore::get(store, left).expect("the row").revision;
            store.remove(left, revision).expect("removed");
            let row = admitted(store, "taker");
            assert_eq!(index_id(store, row), freed, "the id is taken again");
            then(row);
            taker = Some(row);
        });
        assert!(before.contains(&left), "the world of before holds the row");
        (each, taker.expect("a taker"))
    }

    /// The same under one seed.
    fn named_past_an_id_taken_again(
        store: &MemoryStore,
        left: WorkerId,
        request: &[u64],
        (cap, seed): (u32, u64),
        then: impl FnOnce(WorkerId),
    ) -> (Vec<(WorkerId, u32)>, WorkerId) {
        let (mut each, taker) =
            named_past_an_id_taken_again_each(store, left, request, cap, &[seed], then);
        (each.pop().expect("one seed"), taker)
    }

    /// An index id is free for the next row once its row leaves, and a
    /// decision may hold a world published before that. It reads the live
    /// index under that world's tables: the row that left is named no
    /// holder by the blocks of the row that took its id, and a row that
    /// stands is named by what it holds now.
    #[test]
    fn an_index_id_taken_again_names_no_holder_under_the_row_that_left() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        let (left, stays) = (admitted(&store, "left"), admitted(&store, "stays"));
        let request = [1, 2, 3];
        // The taker holds the request whole, and the row that stands comes
        // to hold two of its blocks.
        let (holders, taker) =
            named_past_an_id_taken_again(&store, left, &request, (16, 7), |taker| {
                holds(&store, taker, &request);
                holds(&store, stays, &[1, 2]);
            });
        assert_eq!(holders, [(stays, 2)]);
        // The world of now names both by what they hold.
        let now = handed(&store, GatewayId::LOCAL, None, &request, (16, 7));
        assert_eq!(now.holders.len(), 2);
        assert!(now.holders.contains(&(stays, 2)) && now.holders.contains(&(taker, 3)));
    }

    /// The naming goes on past an id taken again: a row that stands is named
    /// by its depth whether the lookup read it before that id or after it.
    #[test]
    fn a_row_read_after_an_index_id_taken_again_is_still_named() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        // The row that leaves stands between the two that stay, by its id
        // and, once taken again, by its depth.
        let shallow = admitted(&store, "shallow");
        let left = admitted(&store, "left");
        let deep = admitted(&store, "deep");
        let request = [1, 2, 3];
        let (holders, _) = named_past_an_id_taken_again(&store, left, &request, (16, 7), |taker| {
            holds(&store, shallow, &[1]);
            holds(&store, taker, &[1, 2]);
            holds(&store, deep, &request);
        });
        assert_eq!(holders.len(), 2, "{holders:?}");
        assert!(holders.contains(&(shallow, 1)) && holders.contains(&(deep, 3)));
    }

    /// An id taken again takes no place of the cap: the lookup runs again
    /// within the ids not taken since, and the cap is filled from them.
    /// The taker is the deepest of three under a cap of two, so one lookup
    /// alone would name one row.
    #[test]
    fn an_index_id_taken_again_takes_no_place_of_the_cap() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        let left = admitted(&store, "left");
        let (shallow, deep) = (admitted(&store, "shallow"), admitted(&store, "deep"));
        let request = [1, 2, 3];
        let (holders, _) = named_past_an_id_taken_again(&store, left, &request, (2, 7), |taker| {
            holds(&store, taker, &request);
            holds(&store, deep, &[1, 2]);
            holds(&store, shallow, &[1]);
        });
        assert_eq!(holders.len(), 2, "{holders:?}");
        assert!(holders.contains(&(deep, 2)) && holders.contains(&(shallow, 1)));
    }

    /// The lookup made again cuts by a draw of its own. Three rows hold one
    /// block under a cap of one, and one of them is on an id taken again:
    /// where the first lookup drew that id, the second draws between the two
    /// that stand, and across seeds each is named as often as the other. By
    /// the first lookup's stream again the lower id would be named two
    /// times of three.
    #[test]
    fn the_rows_that_stand_beside_an_id_taken_again_are_drawn_evenly() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        let left = admitted(&store, "left");
        let (first, second) = (admitted(&store, "first"), admitted(&store, "second"));
        let request = [1];
        let seeds: Vec<u64> = (0..600).collect();
        let (each, _) =
            named_past_an_id_taken_again_each(&store, left, &request, 1, &seeds, |taker| {
                holds(&store, taker, &request);
                holds(&store, first, &request);
                holds(&store, second, &request);
            });
        let of = |row| {
            each.iter()
                .filter(|holders| **holders == [(row, 1)])
                .count()
        };
        let lower = of(first);
        assert_eq!(
            lower + of(second),
            seeds.len(),
            "one row that stands, each seed"
        );
        // 600 draws of one in two: about 300 a row.
        assert!((240..=360).contains(&lower), "{lower} of 600");
    }

    /// The same under a cap of two, where a lookup cuts by two values: four
    /// rows hold one block and one of them is on an id taken again. Across
    /// seeds each of the three that stand is named two times of three. A
    /// round that read on in the first lookup's stream would cut by that
    /// lookup's second value again, and name the lowest id three times of
    /// four.
    #[test]
    fn two_rows_that_stand_beside_an_id_taken_again_are_drawn_evenly() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        let left = admitted(&store, "left");
        let stand = [
            admitted(&store, "first"),
            admitted(&store, "second"),
            admitted(&store, "third"),
        ];
        let request = [1];
        let seeds: Vec<u64> = (0..2_400).collect();
        let (each, _) =
            named_past_an_id_taken_again_each(&store, left, &request, 2, &seeds, |taker| {
                holds(&store, taker, &request);
                for &row in &stand {
                    holds(&store, row, &request);
                }
            });
        assert!(
            each.iter().all(|holders| holders.len() == 2),
            "two rows, each seed"
        );
        let named = stand.map(|row| {
            let holds = |holders: &&Vec<(WorkerId, u32)>| holders.contains(&(row, 1));
            each.iter().filter(holds).count()
        });
        // 2,400 draws of two in three: about 1,600 a row.
        assert!(
            named.iter().all(|count| (1_500..=1_700).contains(count)),
            "{named:?}"
        );
    }

    /// The ids a lookup made again is kept within are told through every
    /// word of the mask. In a pool whose ids pass the first word, an id
    /// taken again in the second leaves a row that stands on the first
    /// named by what it holds, and an id taken again in the first leaves a
    /// row that stands on the second named.
    #[test]
    fn an_index_id_taken_again_is_told_apart_past_the_masks_first_word() {
        let request = [1, 2, 3];
        // The row that leaves holds the id `at`, and the one named `stays`.
        let named = |at: usize, stays: usize| {
            let store = MemoryStore::new();
            store.apply(sampled(1)).expect("above the unbound start");
            let rows: Vec<WorkerId> = (0..72)
                .map(|n| admitted(&store, &format!("w{n:02}")))
                .collect();
            let (left, stays) = (rows[at], rows[stays]);
            assert_eq!(index_id(&store, left) as usize, at, "ids as admitted");
            let (holders, _) =
                named_past_an_id_taken_again(&store, left, &request, (16, 7), |taker| {
                    holds(&store, taker, &request);
                    holds(&store, stays, &[1, 2]);
                });
            (holders, stays)
        };
        let (holders, stays) = named(70, 6);
        assert_eq!(holders, [(stays, 2)]);
        let (holders, stays) = named(6, 70);
        assert_eq!(holders, [(stays, 2)]);
    }

    /// What a step inside one naming reaches: the store, the pool's rows and
    /// the standing the decision holds, as the kv tables are asked whether
    /// a standing stands, the request, the row whose lane left, the id of
    /// the row between, the rows that hold one block, and as many rows no
    /// pool holds.
    struct Inside {
        store: Arc<MemoryStore>,
        pool: Vec<Candidate>,
        standing: Arc<PoolIndex>,
        request: [u64; 3],
        first: WorkerRow,
        between: WorkerId,
        others: Vec<WorkerId>,
        thirds: Vec<WorkerRow>,
    }

    /// One naming with a step inside it, the lanes moved at the kv tables
    /// as inside one publication window. The rows that stand: `first`,
    /// `others` rows holding one block of the request each, and `stays`
    /// holding two. The lane of `first` leaves, a row between takes its id
    /// and holds the request under it, and a decision that holds the world
    /// of before names the request: at `point` of that naming `step` runs,
    /// and the naming is held to have reached it and to leave no step set
    /// behind it. The holders by their
    /// workers, then the rows that hold one block and `stays`.
    fn named_with_a_step_inside(
        point: seam::Point,
        others: usize,
        step: impl FnOnce(Inside) + 'static,
    ) -> (Vec<(WorkerId, u32)>, Vec<WorkerId>, WorkerId) {
        let store = Arc::new(MemoryStore::new());
        store.apply(sampled(1)).expect("above the unbound start");
        let first = admitted(&store, "first");
        let others: Vec<WorkerId> = (0..others)
            .map(|at| admitted(&store, &format!("other-{at}")))
            .collect();
        let stays = admitted(&store, "stays");
        let request = [1, 2, 3];
        for &other in &others {
            holds(&store, other, &[1]);
        }
        holds(&store, stays, &[1, 2]);
        let row = WorkersStore::get(&*store, first).expect("the row");
        // A row of the same facts under an id of its own: one no pool holds.
        let another = |id: u64, key: &str| {
            let mut row = row.clone();
            row.id = WorkerId(id);
            row.facts.key = key.to_string();
            row.facts.engine = key.to_string();
            row
        };
        let between = another(0xb0, "between");
        let thirds: Vec<WorkerRow> = (0xc0..)
            .take(others.len())
            .map(|id| another(id, &format!("third-{id:x}")))
            .collect();
        let mut view = ViewBuffers::default();
        let mut named = Named::default();
        let rows: Vec<WorkerId> = store.deciding(|world| {
            let ask = Ask {
                gateway: GatewayId::LOCAL,
                now: Instant::now(),
                pinned: None,
                affinity: false,
            };
            let built = world
                .build(MODEL, Role::Single, &ask, &mut view)
                .expect("the pool");
            let standing = world.standing(MODEL, Role::Single).expect("the pool's");
            let kv = store.kv();
            kv.forget(first);
            assert!(kv.member(&between).is_some(), "the row between takes an id");
            let held = kv.admit(between.id, 0, batch(1), &stored(None, &chain(&request)));
            assert_eq!(held, BatchOutcome::Applied);
            let inside = Inside {
                store: Arc::clone(&store),
                pool: view.candidates.clone(),
                standing,
                request,
                first: row.clone(),
                between: between.id,
                others: others.clone(),
                thirds,
            };
            let ran = Rc::new(Cell::new(false));
            let told = Rc::clone(&ran);
            seam::set(point, move || {
                told.set(true);
                step(inside);
            });
            let naming = Naming {
                gateway: GatewayId::LOCAL,
                block_hashes: &request,
                seed: 7,
                cap: 16,
                wires: None,
                now_micros: built.now_micros,
            };
            world.name(MODEL, Role::Single, &naming, &view.candidates, &mut named);
            assert!(ran.get(), "the step ran inside the naming");
            assert!(!seam::pending(), "no step is left set behind the naming");
            view.candidates.iter().map(|row| row.id).collect()
        });
        let holders = named
            .holders
            .iter()
            .map(|holder| (rows[holder.row as usize], holder.overlap_blocks))
            .collect();
        (holders, others, stays)
    }

    /// A keep inside one naming. Behind the naming's lookup, which read the
    /// blocks of the row between under the first row's id, that row leaves,
    /// the first row takes the id again, and the standing is kept: its
    /// tables are whole at the count of that instant. The naming read their
    /// count before its lookup, so the count told later covers none of its
    /// reads: the first row is named by no block of the row between.
    #[test]
    fn a_keep_inside_one_naming_covers_none_of_its_reads() {
        let (holders, others, stays) = named_with_a_step_inside(seam::Point::Looked, 1, |inside| {
            let kv = inside.store.kv();
            kv.forget(inside.between);
            assert!(kv.member(&inside.first).is_some(), "the id is taken again");
            let kept = kv.stands(MODEL, &inside.pool, &inside.standing);
            assert!(kept, "the standing is kept");
        });
        assert_eq!(holders.len(), 2, "{holders:?}");
        assert!(holders.contains(&(others[0], 1)) && holders.contains(&(stays, 2)));
    }

    /// A row back on its own id inside one naming, with no keep: the id was
    /// taken since the tables were whole, by the row between and then by
    /// the first row again, and a take tells a later count for good. The
    /// first row is named by no block of the row between.
    #[test]
    fn a_row_back_on_its_id_inside_one_naming_is_named_by_no_other_rows_blocks() {
        let (holders, others, stays) = named_with_a_step_inside(seam::Point::Looked, 1, |inside| {
            let kv = inside.store.kv();
            kv.forget(inside.between);
            assert!(kv.member(&inside.first).is_some(), "the id is taken again");
        });
        assert_eq!(holders.len(), 2, "{holders:?}");
        assert!(holders.contains(&(others[0], 1)) && holders.contains(&(stays, 2)));
    }

    /// One take ahead of a lookup made again: the lane of the row at `at`
    /// of those that hold one block leaves, and a row no pool holds takes
    /// its id and holds the request under it. Then the same ahead of each
    /// of the next `more` lookups made again, a row on each time.
    fn takes(inside: Inside, at: usize, more: usize) {
        let kv = inside.store.kv();
        kv.forget(inside.others[at]);
        assert!(
            kv.member(&inside.thirds[at]).is_some(),
            "the row takes an id"
        );
        let events = stored(None, &chain(&inside.request));
        let held = kv.admit(inside.thirds[at].id, 0, batch(1), &events);
        assert_eq!(held, BatchOutcome::Applied);
        if more > 0 {
            seam::set(seam::Point::Again, move || takes(inside, at + 1, more - 1));
        }
    }

    /// A second round. Ahead of the lookup made again, the lane of the row
    /// that holds one block leaves, and a third row takes its id and holds
    /// the request under it. That lookup keeps the id, the ask tells it
    /// taken since, and a second lookup names the row that stands.
    #[test]
    fn a_take_ahead_of_a_lookup_made_again_makes_a_second_round() {
        let (holders, _, stays) =
            named_with_a_step_inside(seam::Point::Again, 1, |inside| takes(inside, 0, 0));
        assert_eq!(holders, [(stays, 2)]);
    }

    /// The fourth round. A take ahead of each of three lookups made again
    /// leaves the fourth to name the rows that stand: the last of those
    /// that hold one block, and the row that holds two.
    #[test]
    fn takes_ahead_of_three_lookups_made_again_leave_the_fourth_to_name() {
        let (holders, others, stays) =
            named_with_a_step_inside(seam::Point::Again, 4, |inside| takes(inside, 0, 2));
        assert_eq!(holders.len(), 2, "{holders:?}");
        assert!(holders.contains(&(others[3], 1)) && holders.contains(&(stays, 2)));
    }

    /// The bound. A take ahead of each of the four lookups made again
    /// names nothing, though the row that holds two blocks stands: a naming
    /// follows the takes four rounds and no further, and it hands no row
    /// its last lookup kept.
    #[test]
    fn takes_ahead_of_four_lookups_made_again_name_nothing() {
        let (holders, _, _) =
            named_with_a_step_inside(seam::Point::Again, 4, |inside| takes(inside, 0, 3));
        assert!(holders.is_empty(), "{holders:?}");
    }

    /// A row that joins takes an id of its own, and the index counts that
    /// take as any: a decision that holds the world of before asks the rows
    /// it kept, finds no id of theirs taken since, and names them by its
    /// one lookup. The row that joined is no row of that world.
    #[test]
    fn a_join_under_a_kept_world_names_that_worlds_rows_as_before() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        let (near, far) = (admitted(&store, "near"), admitted(&store, "far"));
        let request = [1, 2, 3];
        holds(&store, near, &request);
        holds(&store, far, &[1]);
        let (holders, before) = named_under_a_kept_world(&store, &request, (16, 7), || {
            let joined = admitted(&store, "joined");
            holds(&store, joined, &request);
        });
        assert_eq!(before.len(), 2, "the world of before holds two rows");
        assert_eq!(holders.len(), 2, "{holders:?}");
        assert!(holders.contains(&(near, 3)) && holders.contains(&(far, 1)));
    }

    /// A row that joins another pool of the model takes an id of the
    /// model's index, and the write that admits it keeps this pool's
    /// standing. It tells the standing whole at the index's new count, so a
    /// decision on it still asks its one question and looks up once.
    #[test]
    fn a_standing_kept_through_another_pools_take_is_whole_at_the_new_count() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        admitted(&store, "a");
        admitted(&store, "b");
        // The pool's standing, the count its tables are whole at, and the
        // index's count of takes.
        let stood = || {
            store.deciding(|world| {
                let standing = world.standing(MODEL, Role::Single).expect("the pool's");
                let whole_at = standing.tables.taken.load(Ordering::Relaxed);
                let index = standing.tables.index.as_ref().expect("an index");
                (Arc::clone(&standing), whole_at, index.taken())
            })
        };
        let (before, whole_at, taken) = stood();
        assert_eq!(whole_at, taken);
        let other = WorkerFacts {
            role: Role::Decode,
            ..publisher("d", &[MODEL])
        };
        registered(&store, other);
        let (after, whole_at, taken_now) = stood();
        assert!(Arc::ptr_eq(&before, &after), "the standing was kept");
        assert_eq!(taken_now, taken + 1, "the other pool's row took an id");
        assert_eq!(whole_at, taken_now);
    }

    /// A refresh keeps the publication window, and one that moves a row to
    /// another first model frees its index id with no publication. Where a
    /// registration takes the id inside that window, the newest world still
    /// lists the row that moved, and no decision names it a holder by the
    /// blocks of the row that took its id. A row that stands is named by
    /// what it holds, and the window's build names the taker.
    #[test]
    fn an_index_id_taken_again_inside_a_window_names_no_holder_under_the_row_that_moved() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        let (moves, stays) = (admitted(&store, "moves"), admitted(&store, "stays"));
        let request = [1, 2, 3];
        // No build is asked between the two writes: the refresh frees the
        // id and the registration takes it, both inside one window.
        let moved = store.upsert(publisher("moves", &["n", MODEL])).id();
        let taker = store.upsert(publisher("taker", &[MODEL])).id();
        assert_eq!(moved, moves, "a refresh keeps the worker's number");
        holds(&store, taker, &request);
        holds(&store, stays, &[1, 2]);
        assert!(store.publish_due().is_some(), "the window holds the writes");
        for cap in [1, 2, 16] {
            let inside = handed(&store, GatewayId::LOCAL, None, &request, (cap, 7));
            assert!(inside.rows.contains(&moves) && !inside.rows.contains(&taker));
            let moved_named = inside.holders.iter().any(|&(id, _)| id == moves);
            assert!(!moved_named, "under a cap of {cap}: {:?}", inside.holders);
            assert_eq!(inside.holders, [(stays, 2)], "under a cap of {cap}");
        }
        store.publish_now();
        let now = handed(&store, GatewayId::LOCAL, None, &request, (16, 7));
        assert_eq!(now.holders.len(), 2, "{:?}", now.holders);
        assert!(now.holders.contains(&(stays, 2)) && now.holders.contains(&(taker, 3)));
    }

    /// The pool's warming rows are named as a publication read them: the
    /// thin once the fleet holds a cache worth catching up to, then the
    /// young, and nobody once the window is over. A write to the index
    /// moves them at the next window, never between two.
    #[test]
    fn the_warming_rows_follow_the_index_and_the_clock_window_by_window() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        let (a, b) = (admitted(&store, "a"), admitted(&store, "b"));
        let c = admitted(&store, "c");
        let t0 = Instant::now();
        let warming = || {
            let got = handed(&store, GatewayId::LOCAL, None, &[], (16, 1));
            (got.warming, got.thin)
        };
        // A window's publication: a load record with its instant.
        let window = |at: Instant| store.record(a, 0, LoadRow::default(), at, LoadSource::Stream);
        // Three rows admitted together are all young: nobody to catch up to.
        assert_eq!(warming(), (vec![], false));
        // Two fill past the level's floor. The third is thin at the next
        // window, and not before it.
        let filled: Vec<u64> = (1..=1_100).collect();
        holds(&store, a, &filled);
        holds(&store, b, &filled);
        assert_eq!(warming(), (vec![], false));
        window(t0 + Duration::from_secs(1));
        assert_eq!(warming(), (vec![c], true));
        // Past half the level it is thin no more, and young still: short of
        // its first blocks, inside its window.
        holds(&store, c, &filled[..600]);
        window(t0 + Duration::from_secs(2));
        assert_eq!(warming(), (vec![c], false));
        // Its window over, nobody warms.
        window(t0 + Duration::from_secs(61));
        assert_eq!(warming(), (vec![], false));
    }

    /// Past the cap the warming rows a decision reads are a draw of the cap
    /// from its seed, each row as likely as another; within the cap every
    /// one is read.
    #[test]
    fn warming_rows_past_the_cap_are_a_draw_of_the_cap_each_row_as_likely() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        let ids: Vec<WorkerId> = (0..24)
            .map(|n| admitted(&store, &format!("w{n:02}")))
            .collect();
        // Fourteen fill past the level's floor: the other ten are thin.
        let filled: Vec<u64> = (1..=1_100).collect();
        for &id in &ids[..14] {
            holds(&store, id, &filled);
        }
        let at = Instant::now() + Duration::from_secs(1);
        store.record(ids[0], 0, LoadRow::default(), at, LoadSource::Stream);
        let mut thin = ids[14..].to_vec();
        thin.sort_unstable();
        let all = handed(&store, GatewayId::LOCAL, None, &[], (16, 1));
        assert_eq!((all.warming, all.thin), (thin.clone(), true));
        // No cap reads every one.
        let uncapped = handed(&store, GatewayId::LOCAL, None, &[], (0, 1));
        assert_eq!(uncapped.warming, thin);
        let mut picked = std::collections::BTreeMap::new();
        for seed in 0..1_000 {
            let cut = handed(&store, GatewayId::LOCAL, None, &[], (4, seed)).warming;
            assert_eq!(cut.len(), 4, "seed {seed}");
            for id in cut {
                *picked.entry(id).or_insert(0u32) += 1;
            }
        }
        // 1,000 draws of 4 in 10: about 400 a row.
        assert_eq!(picked.keys().copied().collect::<Vec<_>>(), thin);
        assert!(
            picked.values().all(|count| (300..=500).contains(count)),
            "{picked:?}"
        );
    }

    /// The cap and the draw are over the warming rows the view holds. A
    /// gateway whose wires take half of a pool's thin rows is named the cap
    /// among its own half, each as likely as another: a draw over the
    /// pool's whole list would spend its places on rows the view lost.
    #[test]
    fn warming_rows_are_capped_among_the_rows_the_view_holds() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        // Fourteen rows that state both wires and fill past the level's
        // floor; ten thin rows, five on gRPC and five on HTTP alone.
        let both = WireModes::from_bits(WireModes::GRPC.bits() | WireModes::HTTP.bits());
        let ids: Vec<WorkerId> = (0..24)
            .map(|n| {
                let wire_modes = match n {
                    0..14 => both,
                    _ if n % 2 == 0 => WireModes::GRPC,
                    _ => WireModes::HTTP,
                };
                let facts = WorkerFacts {
                    wire_modes,
                    ..publisher(&format!("w{n:02}"), &[MODEL])
                };
                registered(&store, facts)
            })
            .collect();
        let filled: Vec<u64> = (1..=1_100).collect();
        for &id in &ids[..14] {
            holds(&store, id, &filled);
        }
        let at = Instant::now() + Duration::from_secs(1);
        store.record(ids[0], 0, LoadRow::default(), at, LoadSource::Stream);
        let mut thin: Vec<WorkerId> = ids[14..].iter().copied().step_by(2).collect();
        thin.sort_unstable();
        let grpc = Some(WireModes::GRPC);
        // Within the cap the view's own five are named, and no other.
        let all = handed(&store, GatewayId::LOCAL, grpc, &[], (16, 1));
        assert_eq!((all.rows.len(), all.warming), (19, thin.clone()));
        // Past it every draw fills the cap from the five.
        let mut picked = std::collections::BTreeMap::new();
        for seed in 0..500 {
            let cut = handed(&store, GatewayId::LOCAL, grpc, &[], (4, seed)).warming;
            assert_eq!(cut.len(), 4, "seed {seed}");
            for id in cut {
                *picked.entry(id).or_insert(0u32) += 1;
            }
        }
        // 500 draws of 4 in 5: about 400 a row.
        assert_eq!(picked.keys().copied().collect::<Vec<_>>(), thin);
        assert!(
            picked.values().all(|count| (350..=450).contains(count)),
            "{picked:?}"
        );
    }

    /// One predicate says what a decision reaches: over a pool of every
    /// kind of wires, with rows local to two gateways, the rows a read is
    /// kept within are the rows the view holds once the decider narrowed
    /// it, for each gateway and each set of wires it may state.
    #[test]
    fn a_read_is_kept_within_the_rows_the_narrowed_view_holds() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        let (first, second) = (GatewayId(1), GatewayId(2));
        // Every set of the three wires a member may state, none among them,
        // each under every reach.
        for bits in 0u8..8 {
            for local in 0u8..3 {
                let facts = WorkerFacts {
                    wire_modes: WireModes::from_bits(bits),
                    reachable_from: match local {
                        0 => Reachability::Any,
                        1 => Reachability::Gateway(first),
                        _ => Reachability::Gateway(second),
                    },
                    ..publisher(&format!("w{bits}-{local}"), &[MODEL])
                };
                registered(&store, facts);
            }
        }
        for gateway in [GatewayId::LOCAL, first, second] {
            for bits in 0u8..8 {
                let wires = (bits != 0).then(|| WireModes::from_bits(bits));
                let view = handed(&store, gateway, wires, &[], (16, 1)).rows;
                let naming = Naming {
                    gateway,
                    block_hashes: &[],
                    seed: 1,
                    cap: 16,
                    wires,
                    now_micros: 0,
                };
                let reached = store.deciding(|world| world.reached(MODEL, Role::Single, &naming));
                assert_eq!(reached, view, "gateway {gateway:?}, wires {bits:#05b}");
                assert!(!view.is_empty(), "every set of wires has a member");
            }
        }
    }

    /// A young row's naming ends at its own instant, with no window
    /// published between: a decision before the end of its window reads
    /// it, and a decision past it reads nobody.
    #[test]
    fn a_young_rows_naming_ends_at_its_own_instant_with_no_window() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        let (a, b) = (admitted(&store, "a"), admitted(&store, "b"));
        let c = admitted(&store, "c");
        let t0 = Instant::now();
        // Two fill past the level's floor; the third past half the level
        // and short of its first blocks: young, and not thin.
        let filled: Vec<u64> = (1..=1_100).collect();
        holds(&store, a, &filled);
        holds(&store, b, &filled);
        holds(&store, c, &filled[..600]);
        // A generation applied reads the pools again. No window is
        // published, here or below.
        store.apply(sampled(2)).expect("a later generation");
        let warming = |at: Instant| {
            let got = handed_at(&store, GatewayId::LOCAL, None, &[], (16, 1), at);
            (got.warming, got.thin)
        };
        assert_eq!(warming(t0), (vec![c], false));
        assert_eq!(warming(t0 + Duration::from_secs(50)), (vec![c], false));
        // Sixty seconds past its admission the row is young no more.
        assert_eq!(warming(t0 + Duration::from_secs(61)), (vec![], false));
    }

    /// The warming rows named at `now_micros` of the store's clock, by
    /// worker.
    fn warming_at(store: &MemoryStore, now_micros: u64) -> Vec<WorkerId> {
        let mut view = ViewBuffers::default();
        let mut named = Named::default();
        store.deciding(|world| {
            let ask = Ask {
                gateway: GatewayId::LOCAL,
                now: Instant::now(),
                pinned: None,
                affinity: false,
            };
            world
                .build(MODEL, Role::Single, &ask, &mut view)
                .expect("the pool");
            let naming = Naming {
                gateway: GatewayId::LOCAL,
                block_hashes: &[],
                seed: 1,
                cap: 16,
                wires: None,
                now_micros,
            };
            world.name(MODEL, Role::Single, &naming, &view.candidates, &mut named);
        });
        let rows = &view.candidates;
        let named = named.warming.iter();
        named.map(|&row| rows[row as usize].id).collect()
    }

    /// The ends are exact. A fleet that is all young is named from its
    /// first row's end: at that very microsecond the row that ends there is
    /// named no more, and each of the rest is named until its own.
    #[test]
    fn a_naming_ends_at_its_rows_own_microsecond() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        let before = Instant::now();
        // Admitted a millisecond apart: three ends, in this order.
        let ids = ["a", "b", "c"].map(|key| {
            let id = admitted(&store, key);
            std::thread::sleep(Duration::from_millis(1));
            id
        });
        let window = Duration::from_secs(60);
        let from = store.micros(before + window);
        let until = store.micros(Instant::now() + window);
        let sorted = |ids: &[WorkerId]| {
            let mut ids = ids.to_vec();
            ids.sort_unstable();
            ids
        };
        // The first instant anybody is named is the first row's end: the
        // other two are named there, and it is not.
        let first = (from..=until)
            .find(|&at| !warming_at(&store, at).is_empty())
            .expect("the fleet is named once its first row is young no more");
        assert_eq!(warming_at(&store, first - 1), Vec::<WorkerId>::new());
        assert_eq!(warming_at(&store, first), sorted(&ids[1..]));
        // The second row is named until its own end, and the third until
        // its own.
        let second = (first..=until)
            .find(|&at| !warming_at(&store, at).contains(&ids[1]))
            .expect("the second row's end");
        assert_eq!(warming_at(&store, second - 1), sorted(&ids[1..]));
        assert_eq!(warming_at(&store, second), vec![ids[2]]);
        let third = (second..=until)
            .find(|&at| warming_at(&store, at).is_empty())
            .expect("the third row's end");
        assert_eq!(warming_at(&store, third - 1), vec![ids[2]]);
        assert!(first < second && second < third);
    }

    /// A row that lists the model after its first name is in the pool and in
    /// no index of the model: the id it holds in its first model's index is
    /// no id of this one. The row that stands at that id here keeps its
    /// blocks to itself, as a holder and by its size.
    #[test]
    fn a_row_indexed_under_another_model_takes_no_id_of_this_one() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        // Three rows of the model fill past the level's floor, each beside a
        // row that lists the model second and stands at the same id of its
        // own first model's index.
        let filled: Vec<u64> = (1..=1_100).collect();
        let (mut own, mut others) = (Vec::new(), Vec::new());
        for n in 0..3 {
            let other = publisher(&format!("x{n}"), &["m-first", MODEL]);
            others.push(registered(&store, other));
            let id = admitted(&store, &format!("w{n}"));
            holds(&store, id, &filled);
            own.push((id, 3));
        }
        own.sort_unstable();
        others.sort_unstable();
        // A window's publication reads the sizes.
        let at = Instant::now() + Duration::from_secs(1);
        store.record(own[0].0, 0, LoadRow::default(), at, LoadSource::Stream);
        let got = handed(&store, GatewayId::LOCAL, None, &[1, 2, 3], (16, 1));
        assert_eq!(got.rows.len(), 6);
        // The holders are the model's own rows, and each row indexed
        // elsewhere reads as empty here: thin beside a fleet that is full.
        assert_eq!(got.holders, own);
        assert_eq!((got.warming, got.thin), (others, true));
    }

    /// A view with no row names nobody, whatever its pool holds.
    #[test]
    fn a_view_with_no_row_names_nobody() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        let id = admitted(&store, "a");
        holds(&store, id, &[1, 2, 3]);
        let mut view = ViewBuffers::default();
        let mut named = Named::default();
        store.deciding(|world| {
            let ask = Ask {
                gateway: GatewayId::LOCAL,
                now: Instant::now(),
                pinned: None,
                affinity: false,
            };
            let built = world
                .build(MODEL, Role::Single, &ask, &mut view)
                .expect("the pool");
            // As a subset hint of 0 leaves it.
            view.candidates.clear();
            let naming = Naming {
                gateway: GatewayId::LOCAL,
                block_hashes: &[1, 2, 3],
                seed: 1,
                cap: 16,
                wires: None,
                now_micros: built.now_micros,
            };
            world.name(MODEL, Role::Single, &naming, &view.candidates, &mut named);
        });
        assert!(named.holders.is_empty() && named.warming.is_empty());
    }

    /// A pool's standing in its model's index, as a world holds it.
    type Standing = Arc<PoolIndex>;

    /// The standings of the case's two pools, and the list they stand in.
    type Standings = (Standing, Standing, Arc<[Option<Standing>]>);

    /// A membership write keeps the standing of every pool it did not move:
    /// the same tables and warming rows are published, and the list itself
    /// where the write moved none. A pool moves with a wire a lane's row
    /// states, with a lane, and with an admission; any other pool stands,
    /// whether or not a lane moved elsewhere.
    #[test]
    fn a_membership_write_keeps_the_standing_of_a_pool_it_did_not_move() {
        const BESIDE: &str = "n";
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        let (a, b) = (admitted(&store, "a"), admitted(&store, "b"));
        // A row with no lane: its signals ask for no events.
        let plain = |incarnation: u64| WorkerFacts {
            signals: Signals::default(),
            incarnation,
            ..publisher("p", &[MODEL])
        };
        registered(&store, plain(1));
        registered(&store, publisher("c", &[BESIDE]));
        let standings = || -> Standings {
            store.deciding(|world| {
                let of = |model| world.standing(model, Role::Single).expect("the pool's");
                (of(MODEL), of(BESIDE), world.standings())
            })
        };
        let reached = |wires: Option<WireModes>| {
            let naming = Naming {
                gateway: GatewayId::LOCAL,
                block_hashes: &[],
                seed: 1,
                cap: 16,
                wires,
                now_micros: 0,
            };
            store.deciding(|world| world.reached(MODEL, Role::Single, &naming))
        };
        // A registration written again as it stands moves no pool: the
        // list is the one published before.
        let first = standings();
        registered(&store, publisher("a", &[MODEL]));
        let again = standings();
        assert!(Arc::ptr_eq(&first.2, &again.2));
        assert_eq!(reached(Some(WireModes::GRPC)), vec![]);
        // A lane's row states a wire: its pool stands again, the other as
        // it did, and a read for that wire is kept within the row.
        registered(
            &store,
            WorkerFacts {
                wire_modes: WireModes::GRPC,
                ..publisher("a", &[MODEL])
            },
        );
        let wired = standings();
        assert!(!Arc::ptr_eq(&again.0, &wired.0) && Arc::ptr_eq(&again.1, &wired.1));
        assert_eq!(reached(Some(WireModes::GRPC)), vec![a]);
        // A lane is retired: its pool stands again with no id for the row,
        // and the other pool is found standing by its own lanes.
        let mut both = vec![a, b];
        both.sort_unstable();
        assert_eq!(reached(None), both);
        registered(
            &store,
            WorkerFacts {
                signals: Signals::default(),
                ..publisher("b", &[MODEL])
            },
        );
        let retired = standings();
        assert!(!Arc::ptr_eq(&wired.0, &retired.0) && Arc::ptr_eq(&wired.1, &retired.1));
        assert_eq!(reached(None), vec![a]);
        // The row asks for events again: a lane is made, and its pool
        // stands again with the row's id.
        registered(&store, publisher("b", &[MODEL]));
        let made = standings();
        assert!(!Arc::ptr_eq(&retired.0, &made.0) && Arc::ptr_eq(&retired.1, &made.1));
        assert_eq!(reached(None), both);
        // A lane is made in the other pool: this one stands as it did.
        registered(&store, publisher("d", &[BESIDE]));
        let joined = standings();
        assert!(Arc::ptr_eq(&made.0, &joined.0) && !Arc::ptr_eq(&made.1, &joined.1));
        // A row with no lane is admitted again: as many rows, no lane
        // moved, and the pool stands again by the admission alone.
        std::thread::sleep(Duration::from_millis(2));
        registered(&store, plain(2));
        let readmitted = standings();
        assert!(!Arc::ptr_eq(&joined.0, &readmitted.0) && Arc::ptr_eq(&joined.1, &readmitted.1));
        // Two workers join the other pool inside one window, and one build
        // pays both writes: it derives whole, this pool stands as it did
        // through it, and the other stands again with both rows.
        store.upsert(publisher("e", &[BESIDE]));
        store.upsert(publisher("f", &[BESIDE]));
        store.publish_now();
        let folded = standings();
        assert!(Arc::ptr_eq(&readmitted.0, &folded.0) && !Arc::ptr_eq(&readmitted.1, &folded.1));
        assert_eq!(folded.1.tables.id_of.len(), 4);
    }

    /// A pool's standing goes with the pool where a write moves its place:
    /// models that come one after another move the pools about, with a lane
    /// and without one, and each pool that was there keeps its standing.
    #[test]
    fn a_pool_keeps_its_standing_where_a_write_moves_its_place() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        admitted(&store, "a");
        let standing = |model: &str| {
            store.deciding(|world| {
                let standing = world.standing(model, Role::Single).expect("the pool's");
                let held = |held: &Option<Standing>| {
                    held.as_ref()
                        .is_some_and(|held| Arc::ptr_eq(held, &standing))
                };
                let place = world.standings().iter().position(held);
                (standing, place.expect("it stands in the list"))
            })
        };
        let mut known = vec![(MODEL.to_string(), standing(MODEL))];
        let mut moved = 0;
        for n in 0..12 {
            let model = format!("model-{n}");
            let mut facts = publisher(&format!("w{n}"), &[model.as_str()]);
            if n % 2 == 1 {
                facts.signals = Signals::default();
            }
            registered(&store, facts);
            for (model, (was, place)) in &mut known {
                let (now, at) = standing(model);
                assert!(Arc::ptr_eq(was, &now), "{model} after model-{n}");
                moved += usize::from(at != *place);
                *place = at;
            }
            known.push((model.clone(), standing(&model)));
        }
        // The case is of pools that moved: their order is the map's.
        assert!(moved > 0);
    }

    /// A kept pool is found at its place among as many pools. A worker that
    /// is the last of one model and comes back as the first of another
    /// takes a pool out and brings one in: the pools are as many, the map
    /// lays them in another order, and each pool that was there is read by
    /// its own standing at the place the write gave it.
    #[test]
    fn a_kept_pool_is_found_at_its_place_among_as_many_pools() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        let standing = |model: &str| {
            store.deciding(|world| {
                let standing = world.standing(model, Role::Single).expect("the pool's");
                let held = |held: &Option<Standing>| {
                    held.as_ref()
                        .is_some_and(|held| Arc::ptr_eq(held, &standing))
                };
                let place = world.standings().iter().position(held);
                (standing, place.expect("it stands in the list"))
            })
        };
        let mut known = Vec::new();
        for n in 0..8 {
            let model = format!("fixed-{n}");
            registered(&store, publisher(&format!("f{n}"), &[model.as_str()]));
            known.push(model);
        }
        registered(&store, publisher("hop", &["hop-0"]));
        let pools = store.deciding(|world| world.standings().len());
        let mut known: Vec<(String, (Standing, usize))> = known
            .into_iter()
            .map(|model| {
                let at = standing(&model);
                (model, at)
            })
            .collect();
        let mut moved = 0;
        for n in 1..=24 {
            let model = format!("hop-{n}");
            registered(&store, publisher("hop", &[model.as_str()]));
            let now = store.deciding(|world| world.standings().len());
            assert_eq!(now, pools, "as many pools after {model}");
            for (model, (was, place)) in &mut known {
                let (now, at) = standing(model);
                assert!(Arc::ptr_eq(was, &now), "{model} after hop-{n}");
                moved += usize::from(at != *place);
                *place = at;
            }
        }
        // The case is of pools that moved while their count stood.
        assert!(moved > 0);
    }

    /// A lane's row admitted again under a new incarnation is a row of its
    /// own id, and its lane takes the index id the one before it held. Of
    /// the incarnations that follow one another, one whose id stands where
    /// its row stood leaves as many rows, the same wires and the same ids
    /// at the same places: its admission alone says the pool moved, and the
    /// pool stands again by it.
    #[test]
    fn a_lanes_row_admitted_again_moves_its_pool_by_the_admission_alone() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        let (mut last, _) = (admitted(&store, "a"), admitted(&store, "b"));
        let standing =
            || store.deciding(|world| world.standing(MODEL, Role::Single).expect("the pool's"));
        let mut held = 0;
        for incarnation in 2..=16 {
            let before = standing();
            std::thread::sleep(Duration::from_millis(2));
            let again = registered(
                &store,
                WorkerFacts {
                    incarnation,
                    ..publisher("a", &[MODEL])
                },
            );
            assert_ne!(again, last);
            last = again;
            let after = standing();
            // An id that stands elsewhere among the rows moves the places.
            if before.tables.id_of != after.tables.id_of {
                continue;
            }
            held += 1;
            assert_ne!(before.tables.admitted, after.tables.admitted);
            assert!(!Arc::ptr_eq(&before, &after));
        }
        // The case is of a row that took its place again.
        assert!(held > 0);
    }

    /// The holders' draw and the warming rows' draw are on streams of their
    /// own. Two rows are the request's only holders and the pool's only
    /// warming rows, under a cap of one: over the seeds each list names
    /// either row beside either row of the other. On one stream the two
    /// draws would read one value, and one list's row would give the
    /// other's.
    #[test]
    fn the_holders_and_the_warming_rows_are_drawn_apart() {
        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        // Two rows fill past the level's floor with blocks of their own.
        let filled: Vec<u64> = (1..=1_100).collect();
        for key in ["big-a", "big-b"] {
            holds(&store, admitted(&store, key), &filled);
        }
        // Two hold the request's one block and nothing more: thin.
        let (x, y) = (admitted(&store, "x"), admitted(&store, "y"));
        holds(&store, x, &[9_001]);
        holds(&store, y, &[9_001]);
        let at = Instant::now() + Duration::from_secs(1);
        store.record(x, 0, LoadRow::default(), at, LoadSource::Stream);
        let mut pairs = std::collections::BTreeSet::new();
        for seed in 0..64 {
            let got = handed(&store, GatewayId::LOCAL, None, &[9_001], (1, seed));
            assert_eq!(
                (got.holders.len(), got.warming.len()),
                (1, 1),
                "seed {seed}"
            );
            pairs.insert((got.holders[0].0, got.warming[0]));
        }
        assert_eq!(pairs.len(), 4, "{pairs:?}");
    }

    /// A generation rules the warming rows by its own stage from its apply,
    /// with no window and no membership write between: one whose profile
    /// holds no warm-up stage names nobody, one that holds it again names
    /// them again, and one whose stage holds another number rules by it.
    #[test]
    fn a_generation_rules_the_warming_rows_from_its_apply() {
        use crate::spec::{Pairing, ScoreSpec, ScoreStage};

        let store = MemoryStore::new();
        store.apply(sampled(1)).expect("above the unbound start");
        let (a, b) = (admitted(&store, "a"), admitted(&store, "b"));
        let c = admitted(&store, "c");
        // Two fill past the level's floor and the third holds under half
        // the level: thin, by the window that reads it.
        let filled: Vec<u64> = (1..=1_100).collect();
        holds(&store, a, &filled);
        holds(&store, b, &filled);
        holds(&store, c, &filled[..400]);
        let at = Instant::now() + Duration::from_secs(1);
        store.record(a, 0, LoadRow::default(), at, LoadSource::Stream);
        let warming = || {
            let got = handed(&store, GatewayId::LOCAL, None, &[], (16, 1));
            (got.warming, got.thin)
        };
        assert_eq!(warming(), (vec![c], true));
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        document.generation = 2;
        let unruled = AppliedSpec::validate(&document).expect("a preset validates");
        store.apply(unruled).expect("a later generation");
        assert_eq!(warming(), (vec![], false));
        store.apply(sampled(3)).expect("a later generation");
        assert_eq!(warming(), (vec![c], true));
        // A stage that calls a quarter of the level thin: 400 blocks are
        // thin no more, and young.
        let mut document = PolicySpec::preset("cache_aware").expect("a preset of this build");
        document.generation = 4;
        for profile in document.profiles.values_mut() {
            for stage in &mut profile.score {
                if let ScoreSpec::Params(ScoreStage::WarmupSlice(stage)) = stage {
                    stage.thin_ratio = Some(0.25);
                }
            }
        }
        let quarter = AppliedSpec::validate(&document).expect("a preset validates");
        store.apply(quarter).expect("a later generation");
        assert_eq!(warming(), (vec![c], false));
        // A generation of another pairing mode derives the membership
        // again and rules the pools that stand by its own stage there:
        // half the level is thin again.
        let mut document = PolicySpec::preset("cache_aware").expect("a preset of this build");
        document.generation = 5;
        document.plans.pd_pair.pairing = Pairing::Strict;
        let strict = AppliedSpec::validate(&document).expect("a preset validates");
        store.apply(strict).expect("a later generation");
        assert_eq!(warming(), (vec![c], true));
        // A generation whose stage holds another number and reads the same
        // rows at its apply rules the next window by its own number: 400
        // blocks are thin under 0.45 of the level as under half of it, and
        // 520 under half alone.
        let mut document = PolicySpec::preset("cache_aware").expect("a preset of this build");
        document.generation = 6;
        document.plans.pd_pair.pairing = Pairing::Strict;
        for profile in document.profiles.values_mut() {
            for stage in &mut profile.score {
                if let ScoreSpec::Params(ScoreStage::WarmupSlice(stage)) = stage {
                    stage.thin_ratio = Some(0.45);
                }
            }
        }
        let nearer = AppliedSpec::validate(&document).expect("a preset validates");
        store.apply(nearer).expect("a later generation");
        assert_eq!(warming(), (vec![c], true));
        let longer = chain(&filled[..520]);
        let more = stored(Some(longer[399].seq_hash), &longer[400..]);
        assert_eq!(store.admit(c, 0, batch(2), &more), BatchOutcome::Applied);
        let next = at + Duration::from_secs(1);
        store.record(a, 0, LoadRow::default(), next, LoadSource::Stream);
        assert_eq!(warming(), (vec![c], false));
    }
}
