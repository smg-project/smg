//! The index's cases: a lane's life with its row, and the store's apply
//! path held to the `kv-index` crate's reference index over streams with
//! holes, clears, restarts, gaps and snapshots.

use std::collections::BTreeMap;

use kv_index::{chain_prefix_hash, ApplyError, GroupKind, ReferenceIndexer};

use super::{
    admit, Blocks, ContentHash, KvEvent, LaneCounts, LaneWriter, SequenceHash, StoredBlock,
    INDEX_WORKERS,
};
use crate::{
    engine::view::WorkerId,
    store::{
        kv_events::{RankCursor, GAP_CLEAR_THRESHOLD},
        BatchFacts, BatchOutcome, KvEventSink, MemoryStore, ResyncReason, Signals, SnapshotChunk,
        WorkerFacts, WorkersStore,
    },
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

/// A membership write that moves no member, no lane, no block size of
/// record and no row's word to record walks no pool: each stands in its
/// model's index as it did. A registration refreshed with the offer it had
/// is such a write, and it stays one under the default source, where the
/// pools hold a record of sent prefixes. A worker more makes a lane, and
/// the pools are walked.
#[test]
fn a_write_that_moves_no_member_walks_no_pool() {
    use crate::document::{AppliedSpec, PolicySpec};

    let store = MemoryStore::new();
    admitted(&store, "a");
    admitted(&store, "b");
    let refresh = |priority| {
        store.upsert(WorkerFacts {
            priority,
            ..publisher("a", &[MODEL])
        });
        store.publish_now();
    };
    let walks = store.pool_walks();
    refresh(7);
    assert_eq!(store.pool_walks(), walks, "every pool stands as it did");
    let preset =
        PolicySpec::preset("cache_aware").and_then(|document| AppliedSpec::validate(&document));
    store
        .apply(preset.expect("the preset validates"))
        .expect("the store takes the preset");
    store.publish_now();
    let walks = store.pool_walks();
    refresh(8);
    assert_eq!(store.pool_walks(), walks, "with a record to keep too");
    admitted(&store, "c");
    assert_eq!(store.pool_walks(), walks + 1, "a lane was made");
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

/// A lent body's write of `blocks` as a chain from the root: the lane's map
/// is taken for the write and put back.
fn write_lent(lane: &mut LaneWriter<'_>, blocks: &[StoredBlock]) {
    let mut map = std::mem::take(lane.map());
    let placed = lane.stored(blocks, None, &mut map);
    *lane.map() = map;
    placed.expect("stored");
}

/// A lent batch runs through the row's cursor as a typed one does. An
/// admitted batch runs its body once, inside the lane's turn, and what the
/// body writes is the row's. A batch the cursor skips or answers with a gap
/// runs no body, and the typed admit goes on from the same cursor and map.
#[test]
fn a_lent_batch_runs_its_body_where_the_cursor_admits_it() {
    let store = MemoryStore::new();
    let a = admitted(&store, "a");
    let blocks = chain(&[1, 2, 3]);
    let mut ran = 0;
    let mut body = |lane: &mut LaneWriter<'_>| {
        ran += 1;
        write_lent(lane, &blocks);
    };
    assert_eq!(
        store.admit_through(a, 0, batch(1), &mut body),
        BatchOutcome::Applied
    );
    // A duplicate at the cursor, a gap past it, another rank and a row the
    // store does not hold.
    assert_eq!(
        store.admit_through(a, 0, batch(1), &mut body),
        BatchOutcome::Skipped
    );
    assert_eq!(
        store.admit_through(a, 0, batch(5), &mut body),
        BatchOutcome::Gap {
            expected: 2,
            received: 5
        }
    );
    assert_eq!(
        store.admit_through(a, 1, batch(2), &mut body),
        BatchOutcome::Skipped
    );
    assert_eq!(
        store.admit_through(WorkerId(u64::MAX), 0, batch(1), &mut body),
        BatchOutcome::Skipped
    );
    assert_eq!(ran, 1);
    assert_eq!(held(&store, MODEL, &[1, 2, 3]), BTreeMap::from([(a, 3)]));
    assert_eq!(store.indexed_blocks(a, 0), 3);

    let evicted = [KvEvent::Removed(vec![blocks[2].seq_hash])];
    assert_eq!(store.admit(a, 0, batch(2), &evicted), BatchOutcome::Applied);
    assert_eq!(held(&store, MODEL, &[1, 2, 3]), BTreeMap::from([(a, 2)]));
}

/// A lent body reads a value that moves each time the store empties the
/// rank: a resync, a restart its cursor reads, the worker reported gone, a
/// typed batch's clear. It stands still between them, and the body reads it
/// over a map that is empty by then.
#[test]
fn a_lent_body_reads_each_emptying_of_its_rank() {
    let store = MemoryStore::new();
    let a = admitted(&store, "a");
    let blocks = chain(&[1, 2, 3]);
    let lent = |sequence, clears| {
        let facts = BatchFacts {
            sequence,
            clears,
            ..BatchFacts::default()
        };
        let mut read = None;
        let outcome = store.admit_through(a, 0, facts, &mut |lane| {
            read = Some((lane.emptied(), lane.map().len()));
            write_lent(lane, &blocks);
        });
        assert_eq!(outcome, BatchOutcome::Applied);
        read.expect("the body ran")
    };
    let (first, held) = lent(10, false);
    assert_eq!(held, 0);
    assert_eq!(lent(11, false), (first, 3), "no emptying between the two");

    store.resync(a, Some(0), ResyncReason::DataLoss);
    let (resynced, held) = lent(20, false);
    assert_eq!(held, 0);
    assert_ne!(resynced, first);
    // A clear under the cursor is a restarted publisher: the cursor empties
    // the rank ahead of the body.
    let (restarted, held) = lent(5, true);
    assert_eq!(held, 0);
    assert_ne!(restarted, resynced);
    store.removed(a);
    let (gone, held) = lent(1, false);
    assert_eq!(held, 0);
    assert_ne!(gone, restarted);
    // A typed batch's clear empties the rank through the same routine.
    assert_eq!(
        store.admit(a, 0, batch(2), &[KvEvent::Cleared]),
        BatchOutcome::Applied
    );
    let (cleared, held) = lent(3, false);
    assert_eq!(held, 0);
    assert_ne!(cleared, gone);
}

/// The store makes a row's lane again under the row's id: here the row
/// stops asking for events and asks again. A lane usually sees one resync,
/// its first snapshot, so two lanes that each counted from nothing would
/// read alike. The new lane reads another value than the old one did, and
/// a body that outlives the lane resets its state at the next batch.
#[test]
fn a_lane_made_again_under_one_row_id_reads_another_value() {
    let store = MemoryStore::new();
    let a = admitted(&store, "a");
    let read = || {
        store.resync(a, Some(0), ResyncReason::Snapshot);
        let mut read = None;
        let outcome = store.admit_through(a, 0, batch(1), &mut |lane| {
            read = Some(lane.emptied());
        });
        assert_eq!(outcome, BatchOutcome::Applied);
        read.expect("the body ran")
    };
    let old = read();

    let quiet = WorkerFacts {
        signals: Signals::default(),
        ..publisher("a", &[MODEL])
    };
    assert_eq!(store.upsert(quiet).id(), a, "the row keeps its id");
    assert_eq!(
        store.admit_through(a, 0, batch(2), &mut |_| {}),
        BatchOutcome::Skipped,
        "a row that asks for no events holds no lane"
    );
    assert_eq!(store.upsert(publisher("a", &[MODEL])).id(), a);
    store.publish_now();
    assert_ne!(read(), old);
}

/// A lent body writes the row's view in the model's group views. The view
/// empties with the rank, and its groups stay. It leaves with the row: the
/// row that comes next finds a view of its own.
#[test]
fn a_rows_group_view_empties_with_its_rank_and_leaves_with_the_row() {
    let store = MemoryStore::new();
    let a = admitted(&store, "a");
    let view_of = |id, sequence| {
        let mut view = None;
        let outcome = store.admit_through(id, 0, batch(sequence), &mut |lane| {
            view = Some(lane.groups());
        });
        assert_eq!(outcome, BatchOutcome::Applied);
        view.expect("the body ran")
    };
    let view = view_of(a, 1);
    view.learn(1, Some(GroupKind::State), 16);
    view.group_stored(1, &[SequenceHash(7)]);
    let blocks = |view: &kv_index::WorkerGroups| view.pending_blocks() + view.retained_blocks();
    assert_eq!(blocks(&view), 1);
    assert!(std::sync::Arc::ptr_eq(&view, &view_of(a, 2)));

    store.resync(a, Some(0), ResyncReason::DataLoss);
    assert_eq!(blocks(&view), 0, "the view empties with the rank");
    assert!(view.knows(1), "its groups stay");

    let successor = store
        .upsert(WorkerFacts {
            incarnation: 2,
            ..publisher("a", &[MODEL])
        })
        .id();
    assert_ne!(successor, a);
    let next = view_of(successor, 1);
    assert!(!std::sync::Arc::ptr_eq(&view, &next));
    assert!(!next.knows(1), "the successor's view is its own");
}

/// A lent body that unwinds leaves its rank half written. The store has
/// the lane start over before the unwind goes on, as where the worker was
/// reported gone: the rank is empty, the cursor reads nothing, and the next
/// body reads another value over an empty map.
#[test]
fn a_lent_body_that_unwinds_starts_its_lane_over() {
    let store = MemoryStore::new();
    let a = admitted(&store, "a");
    let blocks = chain(&[1, 2, 3]);
    let mut first = None;
    let outcome = store.admit_through(a, 0, batch(1), &mut |lane| {
        first = Some(lane.emptied());
        write_lent(lane, &blocks);
    });
    assert_eq!(outcome, BatchOutcome::Applied);
    assert_eq!(store.indexed_blocks(a, 0), 3);
    assert!(store.cursor(a, 0).is_some());

    let more = chain(&[4, 5]);
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        store.admit_through(a, 0, batch(2), &mut |lane| {
            write_lent(lane, &more);
            panic!("the body's rules failed behind a write");
        })
    }));
    assert!(unwound.is_err());
    assert_eq!(store.indexed_blocks(a, 0), 0, "the rank is empty");
    assert!(store.cursor(a, 0).is_none(), "the cursor starts over");

    let mut next = None;
    let outcome = store.admit_through(a, 0, batch(1), &mut |lane| {
        next = Some((lane.emptied(), lane.map().len()));
    });
    assert_eq!(outcome, BatchOutcome::Applied);
    let (next, held) = next.expect("the body ran");
    assert_eq!(held, 0);
    assert_ne!(Some(next), first);
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

/// A snapshot its stream did not finish cannot be trusted past what it
/// showed. The cursor starts over, so the next subscription asks from zero,
/// and the blocks of its chunks stay until the next snapshot's start. Every
/// other reason empties the rank, and a rank that is not the row's resets
/// nothing.
#[test]
fn an_abandoned_snapshot_keeps_its_blocks_and_starts_its_cursor_over() {
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
        store.admit(a, 0, chunk(9), &stored(None, &first)),
        BatchOutcome::Applied
    );
    store.resync(a, Some(1), ResyncReason::DataLoss);
    assert_eq!(store.cursor(a, 0).expect("heard").applied, 9, "rank 1");
    assert_eq!(store.indexed_blocks(a, 0), 3);

    store.resync(a, Some(0), ResyncReason::SnapshotAbandoned);
    assert_eq!(store.cursor(a, 0), Some(Default::default()), "from zero");
    assert_eq!(held(&store, MODEL, &[1, 2, 3]), BTreeMap::from([(a, 3)]));
    assert_eq!(store.resyncs(ResyncReason::SnapshotAbandoned), 1);
    // The stream that follows is taken from wherever it starts, and its
    // snapshot's start replaces what the abandoned one showed.
    store.resync(a, None, ResyncReason::Snapshot);
    assert_eq!(store.indexed_blocks(a, 0), 0);
    assert_eq!(
        store.admit(a, 0, chunk(4), &stored(None, &second)),
        BatchOutcome::Applied
    );
    assert_eq!(held(&store, MODEL, &[7, 8]), BTreeMap::from([(a, 2)]));

    store.resync(a, Some(0), ResyncReason::DataLoss);
    assert_eq!(store.cursor(a, 0), Some(Default::default()), "from zero");
    assert_eq!(store.indexed_blocks(a, 0), 0);
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

/// A pool's standing in its model's index: what a write keeps of it, and
/// what moves it.
mod named {
    use std::{
        sync::{atomic::Ordering, Arc},
        time::Duration,
    };

    use super::{admitted, publisher, MODEL};
    use crate::{
        document::{AppliedSpec, PolicySpec},
        engine::view::{Role, WorkerId},
        store::{index::PoolIndex, MemoryStore, Signals, WorkerFacts, WorkersStore},
    };

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

    /// A pool's standing in its model's index, as a world holds it.
    type Standing = Arc<PoolIndex>;

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
}
