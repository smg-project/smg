//! The `Replicate` stream: what a leader sends a standby from the place its
//! hello names, which calls it refuses, and how a stream ends.

use std::{convert::Infallible, path::Path, time::Duration};

use prost::Message as _;
use smg_grand_central::{
    service::{
        controller::LeasePlace,
        elector::{FileLease, LeaseBackend, LeaseRecord},
        proto::{
            self, replica_frame::Frame, replication_event::Event as Sent, worker_event::Event,
        },
    },
    store::WorkerFacts,
};
use tokio::{sync::mpsc, task::JoinSet};
use tonic::{codegen::tokio_stream::wrappers::ReceiverStream, Code};

use crate::{
    cases, client, count, facts, lays, place, replicate, shared, snapshot, stated, turn,
    with_leader, writes, Standby,
};

/// `google.rpc.Status`, as a client of any gRPC stack decodes a status's
/// details.
#[derive(Clone, PartialEq, prost::Message)]
struct RpcStatus {
    #[prost(int32, tag = "1")]
    code: i32,
    #[prost(string, tag = "2")]
    message: String,
    #[prost(message, repeated, tag = "3")]
    details: Vec<prost_types::Any>,
}

/// The record in the lease file now.
fn stored(lease: &Path) -> LeaseRecord {
    FileLease::new(lease)
        .read()
        .expect("the lease file reads")
        .expect("the lease file holds a record")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_standby_is_sent_the_log_from_its_place_and_restored_from_any_other() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let lease = dir.path().join("lease");
    with_leader(shared(&dir), |leader| async move {
        let id = writes(&leader, 3);
        let base = snapshot(leader.replication).await.expect("a snapshot");
        let term = base.replication_term;
        assert_ne!(term, 0, "the leader's place stands under its term");
        // The run drew where its places start. The take, the registration
        // and three turns of the status stand after that place.
        let start = base.replication_seq - 5;
        assert_ne!(start, 0, "a run draws where its places start");
        let nth = |write: u64| start + write;

        // A standby with nothing: the take, the registration, the turns.
        let mut fresh = replicate(leader.replication, place(0, 0))
            .await
            .expect("a stream");
        let first = fresh.batch().await;
        assert_eq!((first.first_seq, first.term), (nth(1), term));
        assert_eq!(
            cases(&first.events),
            [
                "term",
                "membership",
                "membership",
                "membership",
                "membership"
            ]
        );
        let Some(Sent::Membership(registered)) = &first.events[1].event else {
            panic!("the registration");
        };
        assert_eq!((registered.seq, registered.term), (nth(2), term));
        assert!(matches!(&registered.event, Some(Event::Registered(row)) if row.id == id.0));

        // A standby at the snapshot's place reads what was written since.
        let mut joined = replicate(leader.replication, place(nth(5), term))
            .await
            .expect("a stream");
        turn(&leader, id, 3);
        let next = joined.batch().await;
        assert_eq!((next.first_seq, next.term), (nth(6), term));
        assert_eq!(cases(&next.events), ["membership"]);
        // The first stream reads the same event: one log, two readers.
        assert_eq!(fresh.batch().await.events, next.events);
        // An idle second: the leader says where its head stands.
        let beat = joined.beat().await;
        assert_eq!((beat.first_seq, beat.term), (nth(7), term));

        // A position past the head and a position that names no term: each
        // is told to re-sync and restored on the stream it holds, at the
        // leader's head, with no load row, and reads on from there.
        for lost in [
            place(nth(70), term),
            proto::ReplicaHello {
                last_applied_term: 0,
                ..place(nth(5), term)
            },
        ] {
            let mut stream = replicate(leader.replication, lost.clone())
                .await
                .expect("a stream");
            let restored = stream.restored().await.expect("a snapshot");
            let at = (restored.replication_seq, restored.replication_term);
            assert_eq!(at, (nth(6), term), "{lost:?}");
            assert_eq!(restored.workers.len(), 1, "{lost:?}");
            assert!(restored.loads.is_empty(), "{lost:?}");
            let beat = stream.beat().await;
            assert_eq!((beat.first_seq, beat.term), (nth(7), term));
        }

        // A second hello ends the stream.
        joined
            .frames
            .send(proto::ReplicaFrame {
                frame: Some(Frame::Hello(place(nth(6), term))),
            })
            .await
            .expect("the stream's own channel");
        assert_eq!(joined.ended().await, Some(Code::FailedPrecondition));

        // A standby that saw a later term than this replica leads under.
        let later = proto::ReplicaHello {
            term_seen: term + 1,
            ..place(nth(5), term)
        };
        let Err(deposed) = replicate(leader.replication, later).await else {
            panic!("a standby that saw a later term is not served");
        };
        assert_eq!(deposed.code(), Code::FailedPrecondition);
        // The details are a `google.rpc.Status`, the form every gRPC client
        // decodes them as: the code again, and a NotLeader under its type's
        // URL that names no leader, since this replica knows none.
        let details = RpcStatus::decode(deposed.details()).expect("a google.rpc.Status");
        assert_eq!(details.code, Code::FailedPrecondition as i32);
        let [detail] = details.details.as_slice() else {
            panic!("one detail: {details:?}");
        };
        assert_eq!(
            detail.type_url,
            "type.googleapis.com/smg.policy.v1.NotLeader"
        );
        let named = proto::NotLeader::decode(detail.value.as_slice()).expect("a NotLeader");
        assert_eq!(named.leader_address, "", "this replica knows no leader");

        // The hello's term went to the election, and the leader left its
        // lease: a stream that stood open under its term ends at its next
        // look.
        assert_eq!(fresh.ended().await, Some(Code::Aborted));
        // The leave wrote its release: the lease's record names no holder,
        // under the term the hello stated. The fence closes ahead of that
        // write, so the record is read until it shows, under a timeout of
        // thirty seconds. A fence that ran out would end the stream too,
        // and would leave the holder's record as it stood: the record
        // tells the leave, and not which of the two ended the stream.
        let released = tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                let record = stored(&lease);
                if record.is_released() {
                    break record;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("the leave's release within 30 s");
        assert_eq!(released.term, term + 1);

        // A replica that left its lease serves no standby: the hello it
        // served before the leave is refused now. The case counts on the
        // hold of `settings`: the replica takes no lease again for the
        // lease's duration, long against this case's run, and a call
        // inside that hold meets a closed fence. Such a call is refused
        // ahead of its hello's fields: a place under a later term is
        // judged in the library's cases.
        let Err(left) = replicate(leader.replication, place(nth(5), term)).await else {
            panic!("a replica that left its lease serves no standby");
        };
        assert_eq!(left.code(), Code::FailedPrecondition);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_standby_that_acknowledges_nothing_is_cut_at_its_backlog() {
    let dir = tempfile::tempdir().expect("a temp dir");
    with_leader(shared(&dir), |leader| async move {
        let id = writes(&leader, 0);
        let hello = proto::ReplicaHello {
            max_backlog: 300,
            ..place(0, 0)
        };
        // Rounds of 100 turns of the status, each read whole before the
        // next is written. A standby that acknowledges each round is never
        // 300 behind, were its acknowledgement a whole round late.
        let mut steady = replicate(leader.replication, hello.clone())
            .await
            .expect("a stream");
        let base = snapshot(leader.replication).await.expect("a snapshot");
        let mut head = base.replication_seq;
        for _ in 0..8 {
            for count in 0..100 {
                turn(&leader, id, count);
            }
            head += 100;
            steady.until(head).await;
            steady
                .frames
                .send(proto::ReplicaFrame {
                    frame: Some(Frame::Ack(proto::ReplicationAck { seq: head })),
                })
                .await
                .expect("the stream's own channel");
        }

        // The same log to a standby that acknowledges nothing: sent its 300
        // events and no more, told to re-sync a second later, and restored
        // at the leader's head on the stream it holds.
        let mut slow = replicate(leader.replication, hello)
            .await
            .expect("a stream");
        let mut held = 0;
        let cut = loop {
            let batch = slow.batch().await;
            if batch.resync_required {
                break batch;
            }
            held += batch.events.len();
            assert!(held <= 300, "{held} events and no cut");
        };
        assert_eq!(held, 300, "events before the cut");
        assert!(cut.events.is_empty());
        let restored = slow.snapshot().await.expect("the snapshot");
        assert_eq!(restored.replication_seq, head);
        assert_eq!(slow.beat().await.first_seq, head + 1);

        // A bound under one batch: a standby that acknowledges each batch
        // is sent the same log to its head, 64 events a batch at the most.
        let small = proto::ReplicaHello {
            max_backlog: 64,
            ..place(0, 0)
        };
        let mut patient = replicate(leader.replication, small)
            .await
            .expect("a stream");
        let mut at = 0;
        while at < head {
            let batch = patient.batch().await;
            assert!(!batch.resync_required, "cut at {at}");
            assert!(batch.events.len() <= 64, "{}", batch.events.len());
            at = batch.first_seq + count(batch.events.len()) - 1;
            patient
                .frames
                .send(ack(at))
                .await
                .expect("the stream's own channel");
        }
        assert_eq!(at, head);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_standby_behind_wide_replacements_catches_up_in_messages_the_transport_reads() {
    let dir = tempfile::tempdir().expect("a temp dir");
    with_leader(shared(&dir), |leader| async move {
        // 257 registrations of one worker, each a later incarnation of a row
        // that holds three labels at a value's bound: an event carries the
        // row it replaced beside the row it wrote, and one batch's count of
        // them weighs more than the 4 MiB a client reads.
        for incarnation in 1..=257 {
            let wide = WorkerFacts {
                labels: ["a", "b", "c"]
                    .map(|key| (key.to_owned(), "v".repeat(4_096)))
                    .into(),
                ..facts("http://a:8000", incarnation, 0)
            };
            lays(&leader, wide);
        }
        let mut standby = replicate(leader.replication, place(0, 0))
            .await
            .expect("a stream");
        let base = snapshot(leader.replication).await.expect("a snapshot");
        let events = standby.until(base.replication_seq).await;
        assert_eq!(events.len(), 258);
        let bytes: usize = events.iter().map(prost::Message::encoded_len).sum();
        assert!(bytes > 4 * 1024 * 1024, "{bytes}");
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_controller_that_leads_itself_serves_no_standby() {
    with_leader(LeasePlace::Process, |leader| async move {
        writes(&leader, 3);
        let Err(refused) = replicate(leader.replication, place(0, 0)).await else {
            panic!("no standby follows a lease in the leader's own process");
        };
        assert_eq!(refused.code(), Code::FailedPrecondition);
        let refused = snapshot(leader.replication)
            .await
            .expect_err("nor is one restored from it");
        assert_eq!(refused.code(), Code::FailedPrecondition);
    })
    .await;
}

/// The principal is the transport's, never a field a peer fills in: a hello
/// that names another replica is refused, and so is a call of the scope
/// that states no name.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_standby_is_the_principal_its_call_states() {
    let dir = tempfile::tempdir().expect("a temp dir");
    with_leader(shared(&dir), |leader| async move {
        writes(&leader, 1);
        let other = proto::ReplicaHello {
            replica_id: "replication-2".to_owned(),
            ..place(0, 0)
        };
        let refused = replicate(leader.replication, other)
            .await
            .err()
            .expect("the hello names another replica");
        assert_eq!(refused.code(), Code::Unauthenticated);
        assert_eq!(
            refused.message(),
            "the hello's replica_id is not the principal the call is from"
        );
        let (_frames, sent) = mpsc::channel::<proto::ReplicaFrame>(1);
        let unnamed = client(leader.replication)
            .await
            .replicate(ReceiverStream::new(sent))
            .await
            .expect_err("the call states no name");
        assert_eq!(unnamed.code(), Code::Unauthenticated);
        // The standby its call states joins.
        let mut joined = replicate(leader.replication, place(0, 0))
            .await
            .expect("a stream");
        assert!(!joined.batch().await.events.is_empty());
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_frame_out_of_the_streams_order_is_refused_and_a_frame_of_no_kind_is_invalid() {
    let dir = tempfile::tempdir().expect("a temp dir");
    with_leader(shared(&dir), |leader| async move {
        writes(&leader, 1);
        // The first frame: an ack is out of order, no kind is malformed.
        for (frame, code) in [
            (
                Some(Frame::Ack(proto::ReplicationAck { seq: 1 })),
                Code::FailedPrecondition,
            ),
            (None, Code::InvalidArgument),
        ] {
            let (frames, sent) = mpsc::channel(1);
            frames
                .send(proto::ReplicaFrame { frame })
                .await
                .expect("the stream's own channel");
            let refused = client(leader.replication)
                .await
                .replicate(stated(ReceiverStream::new(sent)))
                .await
                .expect_err("the first frame is the hello");
            assert_eq!(refused.code(), code);
        }
        // After the hello a frame of no kind ends the stream the same way.
        let mut joined = replicate(leader.replication, place(0, 0))
            .await
            .expect("a stream");
        joined
            .frames
            .send(proto::ReplicaFrame { frame: None })
            .await
            .expect("the stream's own channel");
        assert_eq!(joined.ended().await, Some(Code::InvalidArgument));
        // A frame the leader cannot read ends the stream under the
        // transport's own status, never as a clean end: here a frame above
        // the 4 MiB one message may be.
        let mut joined = replicate(leader.replication, place(0, 0))
            .await
            .expect("a stream");
        let wide = proto::ReplicaHello {
            replica_id: "r".repeat(5 << 20),
            ..place(0, 0)
        };
        joined
            .frames
            .send(proto::ReplicaFrame {
                frame: Some(Frame::Hello(wide)),
            })
            .await
            .expect("the stream's own channel");
        assert_eq!(joined.ended().await, Some(Code::OutOfRange));
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_standby_that_closes_its_half_ends_the_stream() {
    let dir = tempfile::tempdir().expect("a temp dir");
    with_leader(shared(&dir), |leader| async move {
        writes(&leader, 1);
        let Standby {
            frames,
            mut batches,
            ..
        } = replicate(leader.replication, place(0, 0))
            .await
            .expect("a stream");
        drop(frames);
        // What was on its way arrives, then the stream ends with no status.
        let ended = tokio::time::timeout(Duration::from_secs(20), async {
            loop {
                match batches.message().await {
                    Ok(Some(_)) => {}
                    Ok(None) => break Ok(()),
                    Err(status) => break Err(status),
                }
            }
        })
        .await
        .expect("the stream ends within 20 s");
        assert!(ended.is_ok(), "{ended:?}");
    })
    .await;
}

/// Removes the lease file at `lease` again and again. A holder's renewal
/// reads the file, then renames a new one over its path: a removal that
/// lands between the two is undone, and the holder never sees its lease
/// gone. A removal that lands outside a renewal stands, and the holder's
/// next look finds no lease.
async fn removed_again(lease: &Path) -> Infallible {
    loop {
        // Gone already, where the holder has not written since.
        let _ = std::fs::remove_file(lease);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// The fence at the wire. A leader whose lease is gone ends an open stream
/// `ABORTED` at its next look of the lease. A call that still waited for
/// its hello is refused too, though the hello names a place the log does
/// not hold: a replica that lost the lead tells no standby to re-sync.
/// That call ends `ABORTED` where it was admitted under the lead, as the
/// feed's own case holds by construction, and at the door otherwise. Both
/// calls are refused after it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_leader_that_loses_its_lease_ends_its_stream_and_joins_no_other() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let lease = dir.path().join("lease");
    with_leader(shared(&dir), |leader| async move {
        writes(&leader, 3);
        let base = snapshot(leader.replication).await.expect("a snapshot");
        let held = place(base.replication_seq, base.replication_term);
        let mut open = replicate(leader.replication, held.clone())
            .await
            .expect("a stream");
        // A second call says no hello yet. The case polls it once before
        // it asks the leader's word below, so its request stands on the
        // connection ahead of that call's: the leader's transport holds
        // the stream before the word is answered. The call's own first
        // step, at the door, is a task of the leader's runtime, and
        // nothing on the wire shows it done.
        let mut waits = client(leader.replication).await;
        let (late, sent) = mpsc::channel(1);
        let mut call = Box::pin({
            let mut waits = waits.clone();
            async move {
                waits
                    .replicate(stated(ReceiverStream::new(sent)))
                    .await
                    .map(|_| ())
            }
        });
        tokio::select! {
            biased;
            early = &mut call => panic!("the call waits for its hello, not {early:?}"),
            () = std::future::ready(()) => {}
        }
        let mut waiting = JoinSet::new();
        waiting.spawn(call);
        waits
            .leader(proto::LeaderRequest::default())
            .await
            .expect("the leader's word while it leads");
        std::fs::remove_file(&lease).expect("the lease file");
        // One removal can be undone by the holder's renewal: the file is
        // removed again until the stream ends.
        let ended = tokio::select! {
            ended = open.ended() => ended,
            never = removed_again(&lease) => match never {},
        };
        assert_eq!(ended, Some(Code::Aborted));
        // The fence is closed, and the hello names a place past the head.
        let lost = place(base.replication_seq + 1_000, base.replication_term);
        late.send(proto::ReplicaFrame {
            frame: Some(Frame::Hello(lost)),
        })
        .await
        .expect("the call's own channel");
        let refused = tokio::time::timeout(Duration::from_secs(20), waiting.join_next())
            .await
            .expect("the leader answers within 20 s")
            .expect("the one call")
            .expect("the call's task")
            .expect_err("a replica that lost the lead joins no feed");
        // Admitted under the lead, the call read the fence again at its
        // hello. Where its first step came after the loss, the door
        // refused it, as it refuses the two calls below.
        assert!(
            matches!(refused.code(), Code::Aborted | Code::FailedPrecondition),
            "{refused:?}"
        );
        for after in [
            replicate(leader.replication, held).await.map(|_| ()),
            snapshot(leader.replication).await.map(|_| ()),
        ] {
            let refused = after.expect_err("no call is served past the lead");
            assert_eq!(refused.code(), Code::FailedPrecondition);
        }
    })
    .await;
}

/// A frame that acknowledges `seq`.
fn ack(seq: u64) -> proto::ReplicaFrame {
    proto::ReplicaFrame {
        frame: Some(Frame::Ack(proto::ReplicationAck { seq })),
    }
}
