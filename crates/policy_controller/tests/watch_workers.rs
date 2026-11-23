//! `WatchWorkers` end to end (design 5.2): this package's binary as a
//! process, two replicas over one lease file. A watcher on the leader's
//! gateway listener learns the fleet an operator registers through the
//! control listener: a snapshot at one place of the log, then every event
//! after it in the log's order, each the event the door answered at the
//! place it answered it. It resumes after the last event it took, and where
//! no log holds that place under its term it is told to start again, never
//! skipped ahead.
//!
//! No stub server: the client is the generated one and the server is the
//! binary. Both listeners are ones their operator named, and every call
//! states its caller's name.
#![expect(
    clippy::expect_used,
    reason = "the case's helpers run inside the test; a failure there is the test's failure"
)]

#[expect(
    dead_code,
    reason = "the failover helpers are the replicas case's and the election cells'"
)]
#[path = "common/replicas.rs"]
mod replicas;

use std::{
    ffi::OsStr,
    net::SocketAddr,
    time::{Duration, Instant},
};

use replicas::{Lease, Record, Replica, ROOM};
use smg_policy_controller::service::{
    elector::ElectionSettings,
    proto::{
        self, policy_controller_client::PolicyControllerClient, register_worker_request::Mode,
        worker_event::Event,
    },
};
use tonic::{metadata::AsciiMetadataValue, transport::Channel, Code, Request, Streaming};

/// One page of a snapshot is 128 rows: this fleet is two.
const FLEET: usize = 130;

/// Long against a loaded host: the leader of the case's start leads until
/// the case ends it. The renew deadline stays well under the kit's room for
/// an agreement, so a leader whose lead opens one deadline after its take
/// is seen in time.
fn settings() -> ElectionSettings {
    ElectionSettings {
        duration: Duration::from_secs(60),
        renew_period: Duration::from_millis(500),
        renew_deadline: Duration::from_secs(10),
    }
}

async fn client(addr: SocketAddr) -> PolicyControllerClient<Channel> {
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("a uri")
        .connect()
        .await
        .expect("the listener accepts");
    PolicyControllerClient::new(channel)
}

/// `message` as a call that states its caller's name.
fn stated<T>(name: &'static str, message: T) -> Request<T> {
    let mut request = Request::new(message);
    request
        .metadata_mut()
        .insert("x-smg-principal", AsciiMetadataValue::from_static(name));
    request
}

/// A watch of `by` at the leader, opened once its store stands in its term.
async fn opened(
    gateway: &mut PolicyControllerClient<Channel>,
    by: &'static str,
    request: proto::WatchWorkersRequest,
) -> Streaming<proto::WorkerEvent> {
    let started = Instant::now();
    loop {
        match gateway.watch_workers(stated(by, request.clone())).await {
            Ok(stream) => return stream.into_inner(),
            Err(status) => {
                assert_eq!(status.code(), Code::Unavailable, "{status:?}");
                assert!(
                    started.elapsed() < Duration::from_secs(10),
                    "the leader's store did not enter its term: {status:?}"
                );
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
    }
}

/// The stream's next event, which the case expects within ten seconds.
async fn next(stream: &mut Streaming<proto::WorkerEvent>) -> proto::WorkerEvent {
    tokio::time::timeout(Duration::from_secs(10), stream.message())
        .await
        .expect("an event within ten seconds")
        .expect("an event, not a status")
        .expect("the stream is open")
}

/// A checked worker nobody dials: it stays pending.
fn spec(url: &str, model: &str) -> proto::WorkerSpec {
    proto::WorkerSpec {
        url: url.to_owned(),
        wire: proto::WireMode::Grpc as i32,
        role: proto::Role::Single as i32,
        models: vec![model.to_owned()],
        ..Default::default()
    }
}

fn add(spec: proto::WorkerSpec) -> Request<proto::RegisterWorkerRequest> {
    let request = proto::RegisterWorkerRequest {
        spec: Some(spec),
        mode: Mode::Add as i32,
        wait_ready: false,
    };
    stated("operator-a", request)
}

/// The row of a registration's answer.
fn row_of(event: &proto::WorkerEvent) -> proto::WorkerRow {
    match event.event.clone() {
        Some(Event::Registered(row)) => Some(row),
        _ => None,
    }
    .expect("a registration's answer")
}

/// A watch that starts after `seq` under `term`.
fn after(seq: u64, term: u64) -> proto::WatchWorkersRequest {
    proto::WatchWorkersRequest {
        resume_seq: seq,
        resume_term: term,
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_watcher_learns_the_fleet_in_the_logs_order_and_resumes_by_its_place() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let lease = dir.path().join("lease");
    let mut replicas: Vec<Replica> = ["replica-a", "replica-b"]
        .iter()
        .map(|id| {
            let named = ["--identity-as-told", "control,gateway"].map(OsStr::new);
            Replica::start_with(id, Lease::File(&lease), settings(), 0, &named).expect("a start")
        })
        .collect();
    let mut record = Record::default();
    let (leader, term) = record.agreed(&mut replicas).await.expect("one leader");
    let follower = 1 - leader;
    let all = proto::WatchWorkersRequest::default;

    // A follower follows no leader's log in this build: it refuses the
    // watch as a standby not yet synced, and names the leader.
    let mut at_follower = client(replicas[follower].gateway).await;
    let refused = at_follower
        .watch_workers(stated("gateway-a", all()))
        .await
        .expect_err("a follower serves no watch");
    assert_eq!(
        (refused.code(), refused.message()),
        (
            Code::Unavailable,
            "this replica does not lead and follows no leader's log"
        )
    );
    let named = replicas::not_leader(&refused);
    assert_eq!(named.leader_address, replicas[leader].gateway.to_string());

    // The call is the gateway scope's, to a caller that states its name.
    let mut gateway = client(replicas[leader].gateway).await;
    let mut operator = client(replicas[leader].control).await;
    let unnamed = gateway
        .watch_workers(all())
        .await
        .expect_err("no principal stands on the call");
    assert_eq!(unnamed.code(), Code::Unauthenticated);
    let denied = operator
        .watch_workers(stated("gateway-a", all()))
        .await
        .expect_err("the control listener mounts no watch");
    assert_eq!(denied.code(), Code::PermissionDenied);

    // An empty fleet is one empty page, the last, under the leader's term.
    let mut first = opened(&mut gateway, "gateway-a", all()).await;
    let page = next(&mut first).await;
    assert_eq!(
        page.event,
        Some(Event::Snapshot(proto::WorkerSnapshot {
            workers: Vec::new(),
            last: true,
        }))
    );
    assert_eq!((page.term, page.membership_epoch), (term, 0));

    // Two pages of workers through the door. Each answer stands at the
    // place after the one before it, and the watcher is told that event at
    // that place.
    let phantom = |n: usize| format!("http://127.0.0.1:9/engine-{n:03}");
    let mut told = Vec::new();
    for n in 0..FLEET {
        let answer = operator
            .register_worker(add(spec(&phantom(n), "model-a")))
            .await
            .expect("registered")
            .into_inner();
        told.push(answer);
    }
    let mut place = page.seq;
    for answer in &told {
        place += 1;
        assert_eq!((answer.seq, answer.term), (place, term));
        assert_eq!(&next(&mut first).await, answer);
    }

    // A watcher that joins now gets the fleet as two pages at one place,
    // the last registration's, by registration key.
    let mut second = opened(&mut gateway, "gateway-a", all()).await;
    let (head, tail) = (next(&mut second).await, next(&mut second).await);
    let epoch = u64::try_from(FLEET).expect("a count");
    for page in [&head, &tail] {
        assert_eq!(
            (page.seq, page.term, page.membership_epoch),
            (place, term, epoch)
        );
    }
    let (Some(Event::Snapshot(head)), Some(Event::Snapshot(tail))) = (head.event, tail.event)
    else {
        panic!("a snapshot is pages");
    };
    assert_eq!(
        (head.workers.len(), head.last, tail.workers.len(), tail.last),
        (128, false, FLEET - 128, true)
    );
    let urls: Vec<String> = head
        .workers
        .iter()
        .chain(&tail.workers)
        .map(|row| row.url.clone())
        .collect();
    assert_eq!(urls, (0..FLEET).map(phantom).collect::<Vec<_>>());

    // A watch of one model holds the rows that serve it.
    let other = "http://127.0.0.1:9/other-model";
    let answer = operator
        .register_worker(add(spec(other, "model-b")))
        .await
        .expect("registered")
        .into_inner();
    for stream in [&mut first, &mut second] {
        assert_eq!(next(stream).await, answer);
    }
    let other_row = row_of(&answer);
    told.push(answer);
    let only = proto::WatchWorkersRequest {
        models: vec!["model-b".to_owned()],
        ..Default::default()
    };
    let mut narrow = opened(&mut gateway, "gateway-a", only).await;
    let page = next(&mut narrow).await;
    let Some(Event::Snapshot(page)) = page.event else {
        panic!("a snapshot first");
    };
    assert!(page.last);
    assert_eq!(
        page.workers
            .iter()
            .map(|row| row.url.as_str())
            .collect::<Vec<_>>(),
        [other]
    );

    // It is told the events of those rows alone: the drain of a worker of
    // another model passes it by.
    let drain = |row: &proto::WorkerRow| {
        let request = proto::DeregisterWorkerRequest {
            worker_id: row.id,
            revision: row.revision,
            drain: true,
        };
        stated("operator-a", request)
    };
    let passed = operator
        .deregister_worker(drain(&row_of(&told[0])))
        .await
        .expect("draining")
        .into_inner();
    let mine = operator
        .deregister_worker(drain(&other_row))
        .await
        .expect("draining")
        .into_inner();
    assert_eq!(
        mine.event,
        Some(Event::Status(proto::StatusChanged {
            worker_id: other_row.id,
            old: proto::WorkerStatus::Pending as i32,
            new: proto::WorkerStatus::Draining as i32,
            health: proto::HealthFlag::Draining as u32,
        }))
    );
    assert_eq!(next(&mut narrow).await, mine);
    for stream in [&mut first, &mut second] {
        assert_eq!(next(stream).await, passed);
        assert_eq!(next(stream).await, mine);
    }
    told.extend([passed, mine]);

    // The revision the registration gave guards the removal still (a status
    // change moves none), and the removal is told to every watch the row
    // was in.
    let removal = proto::DeregisterWorkerRequest {
        worker_id: other_row.id,
        revision: other_row.revision,
        drain: false,
    };
    let removed = operator
        .deregister_worker(stated("operator-a", removal))
        .await
        .expect("removed")
        .into_inner();
    assert!(
        matches!(removed.event, Some(Event::Removed(_))),
        "{removed:?}"
    );
    for stream in [&mut first, &mut second, &mut narrow] {
        assert_eq!(next(stream).await, removed);
    }
    told.push(removed);

    // A watcher resumes after the last event it took: the tail from there
    // and no snapshot.
    let taken = told[99].clone();
    let mut resumed = opened(&mut gateway, "gateway-a", after(taken.seq, taken.term)).await;
    for answer in &told[100..] {
        assert_eq!(&next(&mut resumed).await, answer);
    }

    // Four watches of one caller stand at this replica: its ceiling. One
    // more is refused by name; another caller's watches are its own.
    let refused = gateway
        .watch_workers(stated("gateway-a", all()))
        .await
        .expect_err("one watch past the ceiling");
    assert_eq!(
        (refused.code(), refused.message()),
        (
            Code::ResourceExhausted,
            "this principal holds its ceiling of 4 open watches: end one, then watch again"
        )
    );

    // A place the log does not hold: another term at the position, or a
    // position past the head. One word under no position, then the end.
    let head = told.last().expect("a fleet").seq;
    for lost in [after(taken.seq, term + 1), after(head + 1_000, term)] {
        let mut stream = opened(&mut gateway, "gateway-b", lost).await;
        let word = next(&mut stream).await;
        assert!(
            matches!(word.event, Some(Event::SnapshotRequired(_))),
            "{word:?}"
        );
        assert_eq!((word.seq, word.term), (0, 0));
        assert_eq!(stream.message().await.expect("a clean end"), None);
    }

    // A request this build cannot serve is refused by its field's name.
    let needs = proto::WatchWorkersRequest {
        with_needs: true,
        ..Default::default()
    };
    for (request, field) in [(needs, "with_needs: "), (after(0, term), "resume_term: ")] {
        let refused = gateway
            .watch_workers(stated("gateway-a", request))
            .await
            .expect_err("refused");
        assert_eq!(refused.code(), Code::InvalidArgument);
        assert!(
            refused.message().starts_with(field),
            "{}",
            refused.message()
        );
    }

    // The leader leaves: every open watch ends ABORTED, and the watcher
    // goes where `Leader` sends it.
    replicas[leader].terminate().expect("a signal");
    for stream in [&mut first, &mut second, &mut narrow, &mut resumed] {
        let ended = tokio::time::timeout(Duration::from_secs(20), stream.message())
            .await
            .expect("the stream ends within twenty seconds")
            .expect_err("a status ends the stream");
        assert_eq!(ended.code(), Code::Aborted, "{ended:?}");
    }
    // The follower takes the lease at the leader's release. A release that
    // did not land inside the leader's grace period leaves the lease to run
    // out, and the follower leads once it has: the wait is the lease's own
    // timing and the kit's room.
    let timing = settings();
    let promoted = record
        .promoted(
            &mut replicas,
            term,
            timing.duration + timing.renew_period + ROOM,
        )
        .await
        .expect("the follower leads");

    // The new leader's log is its own. The place the watcher took under the
    // old term is not held there: it is told to start again, and the fleet
    // it then learns is what is registered with this leader (a standby's
    // copy of the rows is replication's step).
    let mut gateway = client(replicas[follower].gateway).await;
    let mut stream = opened(&mut gateway, "gateway-a", after(taken.seq, term)).await;
    let word = next(&mut stream).await;
    assert!(
        matches!(word.event, Some(Event::SnapshotRequired(_))),
        "{word:?}"
    );
    let mut stream = opened(&mut gateway, "gateway-a", all()).await;
    let page = next(&mut stream).await;
    assert_eq!(
        page.event,
        Some(Event::Snapshot(proto::WorkerSnapshot {
            workers: Vec::new(),
            last: true,
        }))
    );
    assert_eq!(page.term, promoted.info.term);
}
