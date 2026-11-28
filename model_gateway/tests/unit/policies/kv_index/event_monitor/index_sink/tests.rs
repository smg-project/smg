use smg_grpc_client::common_proto::{KvBlock, KvBlocksStored, KvCacheEvent, KvSnapshotChunk};
use smg_worker::kv_events::{batch_facts, batch_rank};

use super::*;

/// One stored block of sixteen tokens that start at `first_token`, as
/// batch `sequence` of `rank`.
fn stored(sequence: u64, first_token: u32, rank: Option<i32>) -> KvEventBatch {
    KvEventBatch {
        sequence_number: sequence,
        dp_rank: rank,
        events: vec![KvCacheEvent {
            event_id: sequence,
            data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                blocks: vec![KvBlock {
                    block_hash: i64::from(first_token),
                    token_ids: (first_token..first_token + 16).collect(),
                    block_size: 16,
                    ..Default::default()
                }],
                ..Default::default()
            })),
        }],
        ..Default::default()
    }
}

/// A sink over one fresh index, with a row open for a worker that
/// stands for `rank`.
fn opened(rank: Option<usize>) -> (Arc<KvIndex>, Arc<IndexSink>, OpenRow, RowId) {
    let index = Arc::new(KvIndex::chain());
    let sink = Arc::new(IndexSink::new(Arc::new(DashMap::new())));
    let row = sink
        .open(Arc::clone(&index), "m", "grpc://w1:9000", rank)
        .expect("the index interns the worker");
    let id = row.id;
    (index, sink, row, id)
}

/// Say `batch` to the sink as the collector does.
fn admit(sink: &IndexSink, id: RowId, batch: &KvEventBatch) -> BatchOutcome {
    let rank = batch_rank(batch).expect("the case's rank is a rank");
    sink.admit(id, rank, batch_facts(batch), batch)
}

fn applied(sink: &IndexSink, id: RowId, rank: u16) -> Option<u64> {
    sink.cursor(id, rank).map(|cursor| cursor.applied)
}

/// A worker that left leaves nothing under its id: no cursor, no block,
/// no row. Added again it is a new row that starts from nothing: its
/// first batch is taken wherever it stands, a sequence the old cursor
/// had passed included.
#[test]
fn a_removed_worker_leaves_no_cursor_and_starts_from_nothing_when_added_again() {
    let (index, sink, _row, id) = opened(None);
    assert_eq!(
        admit(&sink, id, &stored(7, 1_000, None)),
        BatchOutcome::Applied
    );
    assert_eq!(
        admit(&sink, id, &stored(8, 2_000, None)),
        BatchOutcome::Applied
    );
    assert_eq!(applied(&sink, id, 0), Some(8));
    assert_ne!(index.current_size(), 0);

    sink.removed(id);
    assert_eq!(sink.cursor(id, 0), None, "no cursor is left under the id");
    assert_eq!(sink.rows.len(), 0);
    assert_eq!(index.current_size(), 0, "its blocks left the index");
    // What is said for the id after the worker left reaches nothing.
    assert_eq!(
        admit(&sink, id, &stored(9, 3_000, None)),
        BatchOutcome::Skipped
    );
    sink.reconnected(id);
    sink.resync(id, None, ResyncReason::DataLoss);
    assert_eq!(index.current_size(), 0);

    let again = sink
        .open(Arc::clone(&index), "m", "grpc://w1:9000", None)
        .expect("the index interns the worker again");
    let next = again.id;
    assert_ne!(next, id);
    assert_eq!(sink.cursor(next, 0), None, "the new row has heard no rank");
    // Sequence 3 is below the old cursor: a row that had kept it would
    // drop the batch as a duplicate.
    assert_eq!(
        admit(&sink, next, &stored(3, 4_000, None)),
        BatchOutcome::Applied
    );
    assert_eq!(applied(&sink, next, 0), Some(3));
}

/// The sink's cursor is what a subscription resumes from. It stands at
/// the last sequence applied across a new connection, shows a replay
/// asked for, and starts over, with the worker's blocks gone, when the
/// server's history is lost.
#[test]
fn the_cursor_stands_across_a_reconnect_and_starts_over_at_a_resync() {
    let (index, sink, _row, id) = opened(None);
    assert_eq!(
        admit(&sink, id, &stored(7, 1_000, None)),
        BatchOutcome::Applied
    );
    assert_eq!(
        admit(&sink, id, &stored(8, 2_000, None)),
        BatchOutcome::Applied
    );
    sink.reconnected(id);
    assert_eq!(applied(&sink, id, 0), Some(8));

    assert_eq!(
        admit(&sink, id, &stored(11, 3_000, None)),
        BatchOutcome::Gap {
            expected: 9,
            received: 11
        }
    );
    assert_eq!(
        sink.cursor(id, 0),
        Some(KvCursor {
            applied: 8,
            replay_pending: true,
            degraded: false
        })
    );

    sink.resync(id, None, ResyncReason::OutOfRange);
    assert_eq!(sink.cursor(id, 0), Some(KvCursor::default()));
    assert_eq!(index.current_size(), 0);
}

/// A resync names a rank as a check: for a rank the row does not stand
/// for, it resets nothing. An abandoned snapshot starts the cursors over
/// and keeps the blocks its chunks brought.
#[test]
fn a_resync_checks_its_rank_and_an_abandoned_snapshot_keeps_its_blocks() {
    let (index, sink, _row, id) = opened(Some(1));
    assert_eq!(
        admit(&sink, id, &stored(4, 1_000, Some(1))),
        BatchOutcome::Applied
    );
    sink.resync(id, Some(0), ResyncReason::DataLoss);
    assert_eq!(applied(&sink, id, 1), Some(4), "rank 0 is not the row's");
    assert_ne!(index.current_size(), 0);

    let chunk = KvEventBatch {
        snapshot: Some(KvSnapshotChunk {
            index: 0,
            count: 2,
            ..Default::default()
        }),
        ..stored(9, 2_000, Some(1))
    };
    assert_eq!(admit(&sink, id, &chunk), BatchOutcome::Applied);
    assert_eq!(applied(&sink, id, 1), Some(9));
    let blocks = index.current_size();
    assert_ne!(blocks, 0);
    sink.resync(id, Some(1), ResyncReason::SnapshotAbandoned);
    assert_eq!(sink.cursor(id, 1), Some(KvCursor::default()));
    assert_eq!(index.current_size(), blocks, "the chunk's blocks stay");

    sink.resync(id, Some(1), ResyncReason::DataLoss);
    assert_eq!(index.current_size(), 0);
}

/// Malformed-snapshot recovery starts every cursor over even when the
/// sink already counted the preceding snapshot as complete.
#[test]
fn an_abandoned_snapshot_resets_cursors_after_the_sink_closed_its_snapshot() {
    let (index, sink, _row, id) = opened(None);
    let chunk = KvEventBatch {
        snapshot: Some(KvSnapshotChunk {
            index: 0,
            count: 1,
            ..Default::default()
        }),
        ..stored(6, 1_000, None)
    };
    assert_eq!(admit(&sink, id, &chunk), BatchOutcome::Applied);
    assert_eq!(applied(&sink, id, 0), Some(6));
    let blocks = index.current_size();
    assert_ne!(blocks, 0);

    sink.resync(id, None, ResyncReason::SnapshotAbandoned);

    assert_eq!(sink.cursor(id, 0), Some(KvCursor::default()));
    assert_eq!(
        index.current_size(),
        blocks,
        "the completed chunk's blocks stay"
    );
}

/// A snapshot's start said ahead of its first chunk changes nothing by
/// itself: this sink clears in the chunk's own batch.
#[test]
fn a_snapshots_start_said_ahead_changes_nothing_by_itself() {
    let (index, sink, _row, id) = opened(None);
    assert_eq!(
        admit(&sink, id, &stored(4, 1_000, None)),
        BatchOutcome::Applied
    );
    let blocks = index.current_size();
    assert_ne!(blocks, 0);
    sink.resync(id, None, ResyncReason::Snapshot);
    assert_eq!(applied(&sink, id, 0), Some(4));
    assert_eq!(index.current_size(), blocks);
}

/// A row dropped without the worker's removal (its task panicked) takes
/// its state with it and leaves the worker's blocks in the index.
#[test]
fn a_row_dropped_without_a_removal_leaves_its_blocks() {
    let (index, sink, row, id) = opened(None);
    assert_eq!(
        admit(&sink, id, &stored(1, 1_000, None)),
        BatchOutcome::Applied
    );
    drop(row);
    assert_eq!(sink.rows.len(), 0);
    assert_eq!(sink.cursor(id, 0), None);
    assert_ne!(index.current_size(), 0);
}

/// The job that says a worker left holds the row itself: a task aborted
/// while the job is pending drops nothing the job needs, and the
/// worker's blocks leave the index. A row the task alone held went with
/// the task, ahead of the job, which then found nothing to remove.
#[test]
fn a_dispatched_removal_holds_its_row_past_its_task() {
    let (index, sink, row, id) = opened(None);
    assert_eq!(
        admit(&sink, id, &stored(1, 1_000, None)),
        BatchOutcome::Applied
    );
    // The collector's hold moves into the job at its dispatch, and the
    // task is gone before the job runs.
    let (feed, id) = row.into_feed();
    std::thread::spawn(move || feed.removed(id))
        .join()
        .expect("the job runs");
    assert_eq!(index.current_size(), 0);
    assert_eq!(sink.rows.len(), 0, "the row went with the job's hold");
}

/// The outcome is read by the cursor. A batch of a rank the row does not
/// index advances that rank's cursor: it is `Applied`, with nothing
/// indexed, and a stored block of it confirms the model's block size. A
/// duplicate below the cursor is `Skipped`.
#[test]
fn the_outcome_says_whether_the_cursor_took_the_batch() {
    let (index, sink, _row, id) = opened(Some(1));
    let other = stored(5, 1_000, Some(0));
    assert_eq!(admit(&sink, id, &other), BatchOutcome::Applied);
    assert_eq!(index.current_size(), 0, "rank 0's block is not the row's");
    assert_eq!(applied(&sink, id, 0), Some(5));
    assert_eq!(sink.block_size("m"), Some(16));
    assert_eq!(admit(&sink, id, &other), BatchOutcome::Skipped);
}

/// A registration's block size stands until a stored block names one,
/// and never replaces a size the events named.
#[test]
fn a_registrations_block_size_is_provisional() {
    let (_index, sink, _row, id) = opened(None);
    assert_eq!(sink.block_size("m"), None);
    sink.set_block_size("m", 64);
    assert_eq!(sink.block_size("m"), Some(64));
    assert_eq!(
        admit(&sink, id, &stored(1, 1_000, None)),
        BatchOutcome::Applied
    );
    assert_eq!(sink.block_size("m"), Some(16));
    sink.set_block_size("m", 64);
    assert_eq!(sink.block_size("m"), Some(16));
}
