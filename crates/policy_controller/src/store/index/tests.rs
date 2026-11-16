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
    store.upsert(publisher(key, &[MODEL])).id()
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
