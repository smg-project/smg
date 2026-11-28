//! A standby restored from a snapshot: the parts on its stream, the log that
//! follows from the snapshot's place, and the snapshot call, which is the
//! operators' read and restores no standby.

use std::{
    net::SocketAddr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use smg_grand_central::service::proto;
use tonic::{Code, Status};

use crate::{
    cases, churn, client, count, facts, lays, place, replicate, shared, snapshot, stated, turn,
    with_leader, writes, written_past, Twin,
};

/// The snapshot call at `addr`, the operators' read.
async fn get_snapshot(addr: SocketAddr) -> Result<proto::Snapshot, Status> {
    client(addr)
        .await
        .get_snapshot(stated(proto::SnapshotRequest::default()))
        .await
        .map(tonic::Response::into_inner)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_snapshot_and_the_batches_after_it_are_the_next_snapshot_under_a_writer() {
    let dir = tempfile::tempdir().expect("a temp dir");
    with_leader(shared(&dir), |leader| async move {
        let stop = Arc::new(AtomicBool::new(false));
        let writer = {
            let hands = (Arc::clone(&leader.workers), Arc::clone(&leader.store));
            let stop = Arc::clone(&stop);
            thread::spawn(move || churn(&hands, &stop))
        };
        // Three standbys restored while the writer writes, each on a stream
        // of its own, which then carries the log from its snapshot's place.
        let mut streams = Vec::new();
        for _ in 0..3 {
            tokio::time::sleep(Duration::from_millis(150)).await;
            let mut standby = replicate(leader.replication, place(1, 0))
                .await
                .expect("a stream");
            let taken = standby.restored().await.expect("a snapshot");
            // The next snapshot parts from this one by a write: the writer
            // is waited for, never taken to have run meanwhile.
            written_past(&leader, taken.replication_seq).await;
            streams.push((standby, taken));
        }
        // The events the case counts are waited for, never timed.
        let first = streams[0].1.replication_seq;
        written_past(&leader, first + 100).await;
        stop.store(true, Ordering::Relaxed);
        let rounds = writer.join().expect("the writer");
        let last = snapshot(leader.replication).await.expect("a snapshot");
        let later = streams.iter().skip(1).map(|(_, taken)| taken.clone());
        let nexts: Vec<_> = later.chain([last]).collect();

        // A snapshot and the batches after it on its stream are the next
        // snapshot: every position after its place, with no gap.
        let mut events = 0;
        for ((mut standby, from), to) in streams.into_iter().zip(nexts) {
            assert!(
                from.replication_seq < to.replication_seq,
                "the writer wrote"
            );
            let mut twin = Twin::of(&from);
            let told = standby.until(to.replication_seq).await;
            let positions = to.replication_seq - from.replication_seq;
            assert_eq!(count(told.len()), positions, "no gap after a snapshot");
            for event in &told {
                twin.apply(event).expect("an event the leader journals");
                events += 1;
            }
            let apart = twin.apart(&Twin::of(&to));
            assert!(
                apart.is_empty(),
                "the snapshot at {} and the events after it are not the snapshot at {}: {apart:?}",
                from.replication_seq,
                to.replication_seq
            );
        }
        assert!(events >= 100, "{events} events of {rounds} rounds");
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_snapshot_call_restores_no_standby() {
    let dir = tempfile::tempdir().expect("a temp dir");
    with_leader(shared(&dir), |leader| async move {
        let id = writes(&leader, 2);
        // A standby's snapshot rides its stream: the row, and no load row.
        let restored = snapshot(leader.replication).await.expect("a snapshot");
        let rows: Vec<u64> = restored.workers.iter().map(|row| row.id).collect();
        assert_eq!(rows, [id.0]);
        assert!(restored.loads.is_empty());
        // The call is the operators' read: the replication listener and the
        // gateways' mount no scope of it. The control scope serves two of
        // its slices, and a request that names none asks for every slice.
        for (addr, code) in [
            (leader.replication, Code::PermissionDenied),
            (leader.gateway, Code::PermissionDenied),
            (leader.control, Code::Unimplemented),
        ] {
            let refused = get_snapshot(addr).await.expect_err("no snapshot");
            assert_eq!(refused.code(), code, "{addr}");
        }
        // The operators' rows over the wire: the row by its id under the
        // leader's term, with no load row and no place of the log. No
        // standby starts from this answer.
        let slice = |slice: proto::snapshot_request::Slice| {
            stated(proto::SnapshotRequest {
                slices: vec![slice.into()],
                ..proto::SnapshotRequest::default()
            })
        };
        let mut operator = client(leader.control).await;
        let read = operator
            .get_snapshot(slice(proto::snapshot_request::Slice::Workers))
            .await
            .expect("the rows")
            .into_inner();
        let rows: Vec<u64> = read.workers.iter().map(|row| row.id).collect();
        assert_eq!(rows, [id.0]);
        assert!(read.term > 0, "{read:?}");
        assert_eq!(read.controller_id, "replication-0");
        assert!(read.loads.is_empty() && read.spec.is_none() && !read.truncated);
        assert_eq!(
            (
                read.replication_seq,
                read.replication_term,
                read.membership_epoch
            ),
            (0, 0, 0)
        );
        // This controller was given no spec source: the spec slice is
        // refused by its name.
        let refused = operator
            .get_snapshot(slice(proto::snapshot_request::Slice::Spec))
            .await
            .expect_err("no spec source");
        assert_eq!(
            (refused.code(), refused.message()),
            (
                Code::Unimplemented,
                "slices: SLICE_SPEC is not served by a controller started with no spec source"
            )
        );
    })
    .await;
}

/// A store whose rows pass one message of the transport restores a standby
/// whole: its snapshot arrives in parts on the standby's stream, each one
/// the transport reads, and the log follows from the snapshot's place.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_store_past_one_message_restores_a_standby_whole_on_its_stream() {
    let dir = tempfile::tempdir().expect("a temp dir");
    with_leader(shared(&dir), |leader| async move {
        // Rows of a few kilobytes each, by their key.
        let wide = "k".repeat(2_000);
        for row in 0..1_500 {
            lays(&leader, facts(&format!("http://{wide}-{row}:8000"), 1, 0));
        }
        let id = writes(&leader, 3);
        let mut standby = replicate(leader.replication, place(1, 0))
            .await
            .expect("a stream");
        let restored = standby.restored().await.expect("a snapshot");
        let (parts, bytes) = (standby.parts.len(), standby.parts.iter().sum::<usize>());
        assert!(bytes > 4 * 1024 * 1024, "{bytes} bytes in {parts} parts");
        // Each part is a page of rows and the batch's own fields at most:
        // half of what the transport reads.
        let page = 2 * 1024 * 1024 + 64;
        assert!(standby.parts.iter().all(|part| *part <= page));
        // Every row the leader holds, at its head, and no load row.
        let held = leader.workers.checkpoint(&leader.store);
        // The cut is whole by the case's own count: the wide rows and the
        // one row of `writes`.
        assert_eq!(held.rows.len(), 1_501, "every row the case laid");
        let at = (restored.replication_seq, restored.replication_term);
        assert_eq!(at, (held.at.seq, held.at.term));
        assert_eq!(restored.membership_epoch, held.epoch);
        let told: Vec<u64> = restored.workers.iter().map(|row| row.id).collect();
        let rows: Vec<u64> = held.rows.iter().map(|row| row.id.0).collect();
        assert_eq!(told, rows);
        // The order is held here: the snapshot was sent from such a cut.
        let keys: Vec<&str> = held.rows.iter().map(|row| &*row.facts.key).collect();
        assert!(keys.windows(2).all(|pair| pair[0] < pair[1]), "by key");
        assert!(restored.loads.is_empty());
        // The log follows from the snapshot's place, on the same stream.
        turn(&leader, id, 3);
        let next = standby.batch().await;
        assert_eq!((next.first_seq, next.term), (at.0 + 1, at.1));
        assert_eq!(cases(&next.events), ["membership"]);
    })
    .await;
}
