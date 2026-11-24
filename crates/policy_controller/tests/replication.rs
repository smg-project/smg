//! The replication scope's call end to end (design 5.4), against the
//! controller's serve in the case's own process: `Replicate` from a
//! standby's place, and the snapshot a standby is restored from on it.
//!
//! No stub server: the client is the generated one and the server is the
//! controller's own serve. The writes go through the handle of its store,
//! the way the controller's own drivers write.
#![expect(
    clippy::expect_used,
    reason = "the case's helpers run inside the test; a failure there is the test's failure"
)]

#[expect(
    dead_code,
    reason = "this binary dials loopback; the other helpers are the listeners' tests'"
)]
#[path = "common/net.rs"]
mod net;
#[path = "common/pair.rs"]
mod pair;

use std::{
    collections::BTreeMap,
    convert::Infallible,
    future::Future,
    io::{ErrorKind, Write as _},
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use net::{http_get, loopback};
use prost::Message as _;
use smg_policy_controller::{
    engine::{LoadRow, WireModes, WorkerId},
    service::{
        controller::{
            Config, Controller, ElectionConfig, Error, LeasePlace, Listener, Readiness, Scope,
            State,
        },
        elector::{ElectionSettings, Elector, FileLease, TickOutcome},
        proto::{
            self, policy_controller_client::PolicyControllerClient, replica_frame::Frame,
            replication_event::Event as Sent, worker_event::Event,
        },
    },
    store::{
        HealthPolicy, LoadSink, LoadSource, MemoryStore, WorkerFacts, WorkerStatus, WorkersStore,
    },
};
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinSet,
};
use tonic::{
    codegen::tokio_stream::wrappers::ReceiverStream, metadata::AsciiMetadataValue,
    transport::Channel, Code, Request, Status, Streaming,
};
use tonic_health::pb::{
    health_check_response::ServingStatus, health_client::HealthClient, HealthCheckRequest,
};

/// Long against a loaded host: the leader of the case's start leads to its
/// end.
fn settings() -> ElectionSettings {
    ElectionSettings {
        duration: Duration::from_secs(60),
        renew_period: Duration::from_millis(500),
        renew_deadline: Duration::from_secs(40),
    }
}

fn listener(name: &str, scopes: &[Scope]) -> Listener {
    Listener {
        name: name.to_owned(),
        addr: loopback(),
        scopes: scopes.to_vec(),
        tls: None,
    }
}

/// What a case reaches of the controller it runs against.
struct Leader {
    replication: SocketAddr,
    control: SocketAddr,
    gateway: SocketAddr,
    store: Arc<MemoryStore>,
}

/// Binds the controller `config` makes of a replication port, at a number
/// of the pairs' own ([`pair::ports`]): the next one free on `ip`, and on
/// `standby` too where the replica leads a pair ([`pair::free`]).
async fn bind_drawn(
    ip: IpAddr,
    standby: Option<IpAddr>,
    config: impl Fn(u16) -> Config,
) -> Controller {
    let mut bound = None;
    for drawn in pair::ports(ip) {
        match Controller::bind(config(drawn)).await {
            // In use on this address: the next number.
            Err(Error::Bind { source, .. }) if source.kind() == ErrorKind::AddrInUse => {}
            // Held on the standby's address by a standby that outlived its
            // leader: the next number, and this bind is let go.
            Ok(_) if standby.is_some_and(|ip| !pair::free(ip, drawn).expect("the look's bind")) => {
            }
            other => {
                bound = Some(other.expect("bind"));
                break;
            }
        }
    }
    bound.expect("a free port among the pairs' own")
}

/// Serves a controller over a lease at `place` in the case's process, and
/// runs `case` against it once it leads.
async fn with_leader<Run>(place: LeasePlace, case: impl FnOnce(Leader) -> Run)
where
    Run: Future<Output = ()>,
{
    // A replica under a shared lease follows, and one that follows names
    // its replication port: a number of the pairs' own.
    let config = |port| Config {
        controller_id: "replication-0".to_owned(),
        listeners: vec![
            listener("gateway", &[Scope::Gateway, Scope::Ingestion]),
            listener("control", &[Scope::Control]),
            Listener {
                addr: SocketAddr::new(LEADER, port),
                ..listener("replication", &[Scope::Replication])
            },
        ],
        admin: loopback(),
        shutdown_settle: Duration::from_millis(50),
        shutdown_grace: Duration::from_secs(5),
        load_poll_interval: Config::DEFAULT_LOAD_POLL_INTERVAL,
        decide_shards: 2,
        discovery: None,
        election: ElectionConfig {
            lease: place.clone(),
            settings: settings(),
            advertise: None,
        },
    };
    let mut controller = bind_drawn(LEADER, None, config).await;
    // Each listener takes an identity as told.
    for name in ["gateway", "control", "replication"] {
        controller
            .identity_as_told(name)
            .expect("a listener of the configuration");
    }
    let addresses = controller.addresses();
    let of = |scope| addresses.of(scope).expect("every scope is mounted");
    let leader = Leader {
        replication: of(Scope::Replication),
        control: of(Scope::Control),
        gateway: of(Scope::Gateway),
        store: controller.store(),
    };
    let mut readiness = controller.readiness();
    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let run = async {
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(30), readiness.wait_for(State::Leader)).await,
            Ok(Some(State::Leader)),
            "the lone replica leads within 30 s"
        );
        case(leader).await;
        stop.send(()).expect("the serve still runs");
    };
    // The serve future is the controller whole: it stands on the heap with
    // the case, so no case's future carries its size.
    let (served, ()) = Box::pin(tokio::time::timeout(Duration::from_secs(120), async {
        tokio::join!(serve, run)
    }))
    .await
    .expect("the case and the drain end within 120 s");
    served.expect("a clean drain");
}

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

/// A lease file of the case's own, which a holder that led released: a
/// shared lease, so a standby can follow its holder, and one the case's
/// replica leads at its take. A take of a record with no holder to wait
/// for opens its fence one renew deadline later.
/// The case's first ask makes the file; a second replica of the case
/// finds it.
fn shared(dir: &tempfile::TempDir) -> LeasePlace {
    let path = dir.path().join("lease");
    if !path.exists() {
        let lease = Box::new(FileLease::new(&path));
        let mut before = Elector::new(lease, "replication-9@127.0.0.1:9", settings())
            .expect("the settings are in order");
        let took = before.tick(Instant::now());
        assert!(matches!(took, Ok(TickOutcome::Acquired { .. })), "{took:?}");
        // A release inside the take's wait hands the wait on, so the holder
        // before leads first: its take opens no fence until a renewal one
        // renew deadline after it.
        let led = Instant::now() + settings().renew_deadline;
        let renewed = before.tick(led);
        assert!(
            matches!(renewed, Ok(TickOutcome::Renewed { .. })),
            "{renewed:?}"
        );
        assert!(before.fence().leader_term(led).is_some(), "{led:?}");
        assert!(before.release().expect("the release's write"));
    }
    LeasePlace::File(path)
}

async fn client(addr: SocketAddr) -> PolicyControllerClient<Channel> {
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("a uri")
        .connect()
        .await
        .expect("the listener accepts");
    PolicyControllerClient::new(channel)
}

/// How long a stream that should end may stand before its case fails:
/// far past any stall of a loaded host, so the wait is for the end
/// itself.
const STANDING: Duration = Duration::from_secs(120);

/// The standby every case is: the name its calls state and its hello's id.
const STANDBY: &str = "replication-1";

/// `message` as a call that states the standby's name.
fn stated<T>(message: T) -> Request<T> {
    let mut request = Request::new(message);
    request
        .metadata_mut()
        .insert("x-smg-principal", AsciiMetadataValue::from_static(STANDBY));
    request
}

/// A standby's open stream: its acknowledgements in, the leader's batches
/// out.
struct Standby {
    frames: mpsc::Sender<proto::ReplicaFrame>,
    batches: Streaming<proto::ReplicationBatch>,
    /// The bytes of each part of the last snapshot read.
    parts: Vec<usize>,
}

fn place(seq: u64, term: u64) -> proto::ReplicaHello {
    proto::ReplicaHello {
        replica_id: STANDBY.to_owned(),
        last_applied_seq: seq,
        term_seen: term,
        max_backlog: 0,
        last_applied_term: term,
    }
}

async fn replicate(addr: SocketAddr, hello: proto::ReplicaHello) -> Result<Standby, Status> {
    let (frames, sent) = mpsc::channel(16);
    frames
        .send(proto::ReplicaFrame {
            frame: Some(Frame::Hello(hello)),
        })
        .await
        .expect("the stream's own channel");
    let batches = client(addr)
        .await
        .replicate(stated(ReceiverStream::new(sent)))
        .await?
        .into_inner();
    Ok(Standby {
        frames,
        batches,
        parts: Vec::new(),
    })
}

impl Standby {
    /// The next batch, the stream's end (`None`) or its status, within 20 s.
    async fn next(&mut self) -> Result<Option<proto::ReplicationBatch>, Status> {
        tokio::time::timeout(Duration::from_secs(20), self.batches.message())
            .await
            .expect("the leader answers within 20 s")
    }

    /// The next frame as it is: a batch of events, the word to re-sync, a
    /// part of a snapshot or a heartbeat.
    async fn frame(&mut self) -> proto::ReplicationBatch {
        self.next()
            .await
            .expect("a batch, not a status")
            .expect("a batch, not the stream's end")
    }

    /// The next batch that carries events or the word to re-sync: a
    /// heartbeat is passed over, five of them at most (a case waits for a
    /// batch that is on its way, never through five silent seconds). A
    /// part of a snapshot fails the case here: a leader sends parts behind
    /// the word alone, and [`Standby::snapshot`] reads them.
    async fn batch(&mut self) -> proto::ReplicationBatch {
        let mut found = None;
        for _ in 0..5 {
            let batch = self.frame().await;
            assert!(batch.snapshot.is_none(), "a part no word announced");
            if batch.resync_required || !batch.events.is_empty() {
                found = Some(batch);
                break;
            }
        }
        found.expect("a batch within five heartbeats")
    }

    /// The next heartbeat, past at most 64 batches. A part of a snapshot
    /// fails the case here, as in [`Standby::batch`].
    async fn beat(&mut self) -> proto::ReplicationBatch {
        let mut found = None;
        for _ in 0..64 {
            let batch = self.frame().await;
            assert!(batch.snapshot.is_none(), "a part no word announced");
            if !batch.resync_required && batch.events.is_empty() {
                found = Some(batch);
                break;
            }
        }
        found.expect("a heartbeat within 64 batches")
    }

    /// The stream's status, past the batches still on their way; `None`
    /// when it ends with none. A leader ends a stream at a look of its own
    /// (at its lease, at its log), and a loaded host stretches the time to
    /// that look past any count of heartbeats: the batches before the end
    /// are read past, however many. A stream that goes on fails here
    /// [`STANDING`] on, or at its 4,096th batch.
    async fn ended(&mut self) -> Option<Code> {
        let started = Instant::now();
        let mut end = None;
        for _ in 0..4_096 {
            match self.next().await {
                Ok(Some(_)) if started.elapsed() < STANDING => {}
                Ok(Some(_)) => break,
                Ok(None) => {
                    end = Some(None);
                    break;
                }
                Err(status) => {
                    end = Some(Some(status.code()));
                    break;
                }
            }
        }
        end.expect("the stream's end within two minutes")
    }

    /// Reads on up to position `seq`: every event after the stream's
    /// start, in order. The tail of a batch that runs past it is left.
    async fn until(&mut self, seq: u64) -> Vec<proto::ReplicationEvent> {
        let mut events = Vec::new();
        let mut next = None;
        loop {
            let batch = self.batch().await;
            assert!(!batch.resync_required, "a batch of events");
            assert_eq!(
                next.unwrap_or(batch.first_seq),
                batch.first_seq,
                "a batch starts where the last ended"
            );
            let last = batch.first_seq + batch.events.len() as u64 - 1;
            next = Some(last + 1);
            events.extend(batch.events);
            if last >= seq {
                let past = usize::try_from(last - seq).expect("a batch's length");
                events.truncate(events.len() - past);
                return events;
            }
        }
    }

    /// The leader's answer to a place it does not hold: the word to
    /// re-sync, then its snapshot on the same stream
    /// ([`Standby::snapshot`]).
    async fn restored(&mut self) -> Result<proto::Snapshot, Status> {
        let word = self.batch().await;
        assert!(word.resync_required, "the word to re-sync");
        assert!(word.events.is_empty() && word.snapshot.is_none());
        assert_eq!((word.first_seq, word.term), (0, 0));
        self.snapshot().await
    }

    /// The parts of the leader's snapshot, read to the last and put
    /// together: the rows, the place every part names and the epoch, and
    /// whatever load rows a part carries (a leader of this build sends
    /// none). The stream stands after it: the log follows from that
    /// place. `parts` holds each part's bytes on the wire.
    async fn snapshot(&mut self) -> Result<proto::Snapshot, Status> {
        let mut whole = proto::Snapshot::default();
        self.parts.clear();
        loop {
            let batch = self.next().await?.expect("a part, not the stream's end");
            let bytes = batch.encoded_len();
            assert!(batch.events.is_empty() && !batch.resync_required);
            let part = batch.snapshot.expect("a part of the snapshot");
            assert_eq!(part.part as usize, self.parts.len(), "the parts in order");
            let at = (batch.first_seq - 1, batch.term, part.membership_epoch);
            if self.parts.is_empty() {
                whole.replication_seq = at.0;
                whole.replication_term = at.1;
                whole.membership_epoch = at.2;
            }
            let said = (
                whole.replication_seq,
                whole.replication_term,
                whole.membership_epoch,
            );
            assert_eq!(said, at, "every part at the snapshot's place");
            self.parts.push(bytes);
            whole.workers.extend(part.workers);
            whole.loads.extend(part.loads);
            if part.last {
                return Ok(whole);
            }
        }
    }
}

/// A worker's facts, its checks off. No worker of the suite answers, and
/// a served controller probes each worker whose checks are on: a probe's
/// verdict would write the store beside a case that counts its writes.
fn facts(key: &str, incarnation: u64, max_running: u32) -> WorkerFacts {
    WorkerFacts {
        key: key.to_owned(),
        engine: key.to_owned(),
        incarnation,
        models: vec!["model-a".to_owned()],
        dp_size: 1,
        max_running: (max_running != 0).then_some(max_running),
        health: HealthPolicy {
            probes: false,
            ..HealthPolicy::default()
        },
        ..WorkerFacts::default()
    }
}

fn running(count: u32) -> LoadRow {
    LoadRow {
        running: count,
        ..LoadRow::default()
    }
}

/// One worker and `more` watched writes of it: its status, turned each
/// time.
fn writes(store: &MemoryStore, more: u32) -> WorkerId {
    let id = store.upsert(facts("http://a:8000", 1, 0)).id();
    for count in 0..more {
        turn(store, id, count);
    }
    id
}

/// One watched write of the worker: its status, ready at an even `count`
/// and not ready at an odd one.
fn turn(store: &MemoryStore, id: WorkerId, count: u32) {
    let revision = WorkersStore::get(store, id).expect("the row").revision;
    let status = if count.is_multiple_of(2) {
        WorkerStatus::Ready
    } else {
        WorkerStatus::NotReady
    };
    store.set_status(id, status, revision);
}

/// The names of a batch's events, in order.
fn cases(events: &[proto::ReplicationEvent]) -> Vec<&'static str> {
    events
        .iter()
        .map(|event| match &event.event {
            Some(Sent::Membership(_)) => "membership",
            Some(Sent::Load(_)) => "load",
            Some(Sent::TermStart(_)) => "term",
            Some(_) => "another",
            None => "unset",
        })
        .collect()
}

/// The leader's snapshot at `addr` as a standby that asks to be restored is
/// sent it (a number under no term, which no log holds): the parts of one
/// stream, put together.
async fn snapshot(addr: SocketAddr) -> Result<proto::Snapshot, Status> {
    replicate(addr, place(1, 0)).await?.restored().await
}

/// The snapshot call at `addr`, the operators' read.
async fn get_snapshot(addr: SocketAddr) -> Result<proto::Snapshot, Status> {
    client(addr)
        .await
        .get_snapshot(stated(proto::SnapshotRequest::default()))
        .await
        .map(tonic::Response::into_inner)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_standby_is_sent_the_log_from_its_place_and_restored_from_any_other() {
    let dir = tempfile::tempdir().expect("a temp dir");
    with_leader(shared(&dir), |leader| async move {
        let id = writes(&leader.store, 3);
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
        turn(&leader.store, id, 3);
        let next = joined.batch().await;
        assert_eq!((next.first_seq, next.term), (nth(6), term));
        assert_eq!(cases(&next.events), ["membership"]);
        // The first stream reads the same event: one log, two readers.
        assert_eq!(fresh.batch().await.events, next.events);
        // A load row on the leader's own table is journaled nowhere and
        // sent to no standby: the head stands where the status left it.
        leader
            .store
            .record(id, 0, running(9), Instant::now(), LoadSource::Stream);
        // An idle second: the leader says where its head stands.
        let beat = joined.beat().await;
        assert_eq!((beat.first_seq, beat.term), (nth(7), term));

        // A position past the head and a position that names no term: each
        // is told to re-sync and restored on the stream it holds, at the
        // leader's head, with no load row of the leader's table, and reads
        // on from there.
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

        // So is one whose place stands under a later term, whatever floor
        // it states: a deposed leader restores no standby.
        let ahead = proto::ReplicaHello {
            term_seen: term,
            ..place(nth(5), term + 1)
        };
        let Err(deposed) = replicate(leader.replication, ahead).await else {
            panic!("a standby ahead of this replica's term is not served");
        };
        assert_eq!(deposed.code(), Code::FailedPrecondition);

        // A second hello ends the stream.
        joined
            .frames
            .send(proto::ReplicaFrame {
                frame: Some(Frame::Hello(place(nth(6), term))),
            })
            .await
            .expect("the stream's own channel");
        assert_eq!(joined.ended().await, Some(Code::FailedPrecondition));
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_standby_that_acknowledges_nothing_is_cut_at_its_backlog() {
    let dir = tempfile::tempdir().expect("a temp dir");
    with_leader(shared(&dir), |leader| async move {
        let id = writes(&leader.store, 0);
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
                turn(&leader.store, id, count);
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
            leader.store.upsert(WorkerFacts {
                labels: ["a", "b", "c"]
                    .map(|key| (key.to_owned(), "v".repeat(4_096)))
                    .into(),
                ..facts("http://a:8000", incarnation, 0)
            });
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

/// What a standby holds of a snapshot and the events after it: the rows,
/// under the store's own rules (a status write moves no revision). No load
/// row is among them: a snapshot that carries one and an event that tells
/// one are no leader's of this build.
#[derive(Debug, Default, PartialEq)]
struct Twin {
    rows: BTreeMap<u64, proto::WorkerRow>,
    /// The membership epoch: the snapshot's, then each membership event's.
    epoch: u64,
}

/// What a case's writer changes of a row, in short.
fn brief(row: &proto::WorkerRow) -> String {
    format!(
        "incarnation {} status {} health {} revision {} max_running {}",
        row.incarnation, row.status, row.health, row.revision, row.max_running
    )
}

impl Twin {
    fn of(snapshot: &proto::Snapshot) -> Self {
        assert!(snapshot.loads.is_empty(), "a snapshot with a load row");
        Self {
            // A row's admission age is each send's own.
            rows: snapshot
                .workers
                .iter()
                .map(|row| {
                    let row = proto::WorkerRow {
                        admitted_age_ms: 0,
                        ..row.clone()
                    };
                    (row.id, row)
                })
                .collect(),
            epoch: snapshot.membership_epoch,
        }
    }

    /// What sets this twin apart from `other`, in short.
    fn apart(&self, other: &Self) -> Vec<String> {
        let mut apart = Vec::new();
        if self.epoch != other.epoch {
            apart.push(format!("epoch {} here, {} there", self.epoch, other.epoch));
        }
        for (id, row) in &self.rows {
            match other.rows.get(id) {
                None => apart.push(format!("row {id} only here")),
                Some(theirs) if theirs != row => apart.push(format!(
                    "row {id}: here {}, there {}",
                    brief(row),
                    brief(theirs)
                )),
                Some(_) => {}
            }
        }
        for id in other.rows.keys() {
            if !self.rows.contains_key(id) {
                apart.push(format!("row {id} only there"));
            }
        }
        apart
    }

    fn apply(&mut self, event: &proto::ReplicationEvent) -> Result<(), String> {
        // Each membership event is the next epoch after the last.
        if let Some(Sent::Membership(told)) = &event.event {
            if told.membership_epoch != self.epoch + 1 {
                return Err(format!(
                    "epoch {} after {}",
                    told.membership_epoch, self.epoch
                ));
            }
            self.epoch = told.membership_epoch;
        }
        match event.event.as_ref().expect("an event this build knows") {
            Sent::Membership(told) => match told.event.as_ref().expect("a case") {
                Event::Registered(row) | Event::Relabelled(row) => {
                    self.rows.insert(row.id, row.clone());
                }
                Event::Replaced(replaced) => {
                    let old = replaced.old.as_ref().expect("the old row");
                    self.rows.remove(&old.id);
                    let new = replaced.new.clone().expect("the new row");
                    self.rows.insert(new.id, new);
                }
                Event::Removed(removed) => {
                    self.rows.remove(&removed.worker_id);
                }
                Event::Status(status) => {
                    let row = self
                        .rows
                        .get_mut(&status.worker_id)
                        .expect("a status of a row the standby holds");
                    row.status = status.new;
                    row.health = status.health;
                }
                other => return Err(format!("no membership event: {other:?}")),
            },
            Sent::TermStart(_) => {}
            // A load row is among them: a leader of this build tells none.
            other => return Err(format!("a kind the leader does not journal: {other:?}")),
        }
        Ok(())
    }
}

/// Registers, flips, refreshes, replaces and removes workers over a few
/// keys and writes their load rows, until `stop`. A load row is a write of
/// the store's own table: it moves no place of the log.
fn churn(store: &MemoryStore, stop: &AtomicBool) -> u32 {
    let mut state = 7u64;
    let mut draw = |below: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % below
    };
    let mut lives = [0u64; 4];
    let mut rounds = 0;
    while !stop.load(Ordering::Relaxed) {
        rounds += 1;
        let index = draw(4) as usize;
        let name = format!("http://w{index}:8000");
        match (store.by_key(&name), draw(8)) {
            (None, _) | (Some(_), 0) => {
                lives[index] += 1;
                store.upsert(facts(&name, lives[index], 0));
            }
            (Some(row), 1) => {
                store.remove(row.id, row.revision);
            }
            (Some(row), 2) => {
                store.upsert(facts(
                    &name,
                    lives[index],
                    row.facts.max_running.unwrap_or(0) + 1,
                ));
            }
            (Some(row), 3) => {
                let status = if row.status == WorkerStatus::Ready {
                    WorkerStatus::NotReady
                } else {
                    WorkerStatus::Ready
                };
                store.set_status(row.id, status, row.revision);
            }
            (Some(row), _) => {
                let rank = draw(2) as u16;
                store.record(
                    row.id,
                    rank,
                    running(rounds),
                    Instant::now(),
                    LoadSource::Stream,
                );
            }
        }
        thread::sleep(Duration::from_micros(100));
    }
    rounds
}

/// Waits, within 20 s, until `store` wrote past position `seq`.
async fn written_past(store: &MemoryStore, seq: u64) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while store.checkpoint().at.seq <= seq {
        assert!(Instant::now() < deadline, "no write past {seq} in 20 s");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_snapshot_and_the_batches_after_it_are_the_next_snapshot_under_a_writer() {
    let dir = tempfile::tempdir().expect("a temp dir");
    with_leader(shared(&dir), |leader| async move {
        let stop = Arc::new(AtomicBool::new(false));
        let writer = {
            let (store, stop) = (Arc::clone(&leader.store), Arc::clone(&stop));
            thread::spawn(move || churn(&store, &stop))
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
            written_past(&leader.store, taken.replication_seq).await;
            streams.push((standby, taken));
        }
        // The events the case counts are waited for, never timed: a load
        // row of the writer is no event of the log.
        let first = streams[0].1.replication_seq;
        written_past(&leader.store, first + 100).await;
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
async fn a_controller_that_leads_itself_serves_no_standby() {
    with_leader(LeasePlace::Process, |leader| async move {
        writes(&leader.store, 3);
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_snapshot_call_restores_no_standby() {
    let dir = tempfile::tempdir().expect("a temp dir");
    with_leader(shared(&dir), |leader| async move {
        let id = writes(&leader.store, 2);
        leader
            .store
            .record(id, 0, running(1), Instant::now(), LoadSource::Stream);
        // A standby's snapshot rides its stream: the row, and no load row
        // of the leader's table.
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
            leader
                .store
                .upsert(facts(&format!("http://{wide}-{row}:8000"), 1, 0));
        }
        let id = writes(&leader.store, 3);
        leader
            .store
            .record(id, 0, running(7), Instant::now(), LoadSource::Stream);
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
        // Every row the leader holds, at its head, and no load row of its
        // table.
        let held = leader.store.checkpoint();
        let at = (restored.replication_seq, restored.replication_term);
        assert_eq!(at, (held.at.seq, held.at.term));
        assert_eq!(restored.membership_epoch, held.epoch);
        let told: Vec<u64> = restored.workers.iter().map(|row| row.id).collect();
        let rows: Vec<u64> = held.rows.iter().map(|row| row.id.0).collect();
        assert_eq!(told, rows);
        assert!(restored.loads.is_empty());
        // The log follows from the snapshot's place, on the same stream.
        turn(&leader.store, id, 3);
        let next = standby.batch().await;
        assert_eq!((next.first_seq, next.term), (at.0 + 1, at.1));
        assert_eq!(cases(&next.events), ["membership"]);
    })
    .await;
}

/// A worker's facts of a production row's shape: a 53-byte address, one
/// 36-byte model, four labels, a zone.
fn wide(worker: u32) -> WorkerFacts {
    let address = format!("http://engine-{worker:04}.serving-pool.example.internal:8000");
    let labels = (0..4)
        .map(|n| (format!("label-key-{n:02}"), format!("value-{worker:04}")))
        .collect();
    WorkerFacts {
        models: vec!["organization/model-name-70b-instruct".to_owned()],
        labels,
        zone: Some("region-1-zone-a".to_owned()),
        engine_kind: "engine-kind".to_owned(),
        ..facts(&address, 1, 64)
    }
}

/// A measurement, run by name with `--ignored --nocapture`: a standby's
/// snapshot at the design's top fleet and at 4,096 workers of a production
/// row's shape (a 53-byte address, one 36-byte model, four labels, a
/// zone). One line: the parts, their bytes together and the widest one,
/// and how long a standby that asks waits for the whole (the leader's
/// cut, its parts, the client's read), with the CPU of both ends over the
/// eleven snapshots timed.
/// Then the leader's memory for a held cut, which is what a watch's kept
/// cut holds: the resident growth of 32 streams left inside their
/// snapshots at 32 places. Each fleet is measured in a process of its
/// own: a resident reading starts where no cut of another fleet was let
/// go.
async fn snapshots_numbers(workers: u32) {
    let dir = tempfile::tempdir().expect("a temp dir");
    with_leader(shared(&dir), |leader| async move {
        let mut ids = Vec::new();
        for worker in 0..workers {
            ids.push(leader.store.upsert(wide(worker)).id());
        }
        let mut out = std::io::stdout();
        let mut micros = Vec::new();
        let mut parts = Vec::new();
        let cpu = cpu_ms();
        for _ in 0..11 {
            let began = Instant::now();
            let mut standby = replicate(leader.replication, place(1, 0))
                .await
                .expect("a stream");
            standby.restored().await.expect("a snapshot");
            micros.push(began.elapsed().as_micros());
            parts = standby.parts;
        }
        let cpu = cpu_ms() - cpu;
        micros.sort_unstable();
        writeln!(
            out,
            "SNAPSHOT workers={workers} parts={} bytes={} widest_part={} median_us={} cpu_ms_of_11={cpu}",
            parts.len(),
            parts.iter().sum::<usize>(),
            parts.iter().max().copied().unwrap_or(0),
            micros[5]
        )
        .expect("stdout");
        // Streams left inside their snapshots, each at a place of its own:
        // a worker turned ready between two moves the head, so no two share
        // a cut. Each read the word and its first part, and no further.
        let mut held = Vec::new();
        let before = resident_kb();
        for id in ids.iter().take(32) {
            turn(&leader.store, *id, 0);
            let mut standby = replicate(leader.replication, place(1, 0))
                .await
                .expect("a stream");
            assert!(standby.batch().await.resync_required);
            assert!(standby.frame().await.snapshot.is_some());
            held.push(standby);
        }
        let grew = resident_kb().saturating_sub(before);
        writeln!(
            out,
            "HELD workers={workers} cuts=32 resident_kb_a_cut={}",
            grew / 32
        )
        .expect("stdout");
        assert_eq!(held.len(), 32, "each stream still holds its cut");
    })
    .await;
}

/// A measurement, run by name with `--ignored --nocapture`: the operators'
/// snapshot of the worker rows at a fleet of a production row's shape (a
/// 53-byte address, one 36-byte model, four labels, a zone), asked whole
/// and under a limit of one row. Each line: the rows told and whether a
/// bound cut them, the answer's bytes, how long the call takes over the
/// wire (the store's cut, the rows' build, the client's read), and how long
/// the store's cut takes alone: what a membership write waits behind a
/// snapshot that copies every row, as a call with no limit and no models
/// does, and a call let in beside another. A lone call under a limit
/// copies its limit's worth alone, and a third line asks one row of the
/// widest list of models a call may state: each row's model is searched
/// in that list under the membership lock. A last line, with a watch
/// open: one row asked eleven times, where each call shares the cut the
/// feed keeps and copies nothing.
async fn operators_snapshot_numbers(workers: u32) {
    let dir = tempfile::tempdir().expect("a temp dir");
    with_leader(shared(&dir), |leader| async move {
        for worker in 0..workers {
            leader.store.upsert(wide(worker));
        }
        let mut cuts = Vec::new();
        for _ in 0..11 {
            let began = Instant::now();
            let cut = leader.store.checkpoint();
            cuts.push(began.elapsed().as_micros());
            assert_eq!(cut.rows.len(), workers as usize);
        }
        cuts.sort_unstable();
        let mut out = std::io::stdout();
        let mut operator = client(leader.control).await;
        // The widest list a call may state: 127 models no row serves, then
        // the one every row serves.
        let widest: Vec<String> = (0..127)
            .map(|n| format!("organization/model-name-{n:03}-instruct"))
            .chain(["organization/model-name-70b-instruct".to_owned()])
            .collect();
        for (limit, models) in [(0u32, Vec::new()), (1, Vec::new()), (1, widest)] {
            let asked = models.len();
            let mut micros = Vec::new();
            let mut told = (0, false, 0);
            for _ in 0..11 {
                let request = stated(proto::SnapshotRequest {
                    slices: vec![proto::snapshot_request::Slice::Workers.into()],
                    limit,
                    models: models.clone(),
                    ..proto::SnapshotRequest::default()
                });
                let began = Instant::now();
                let answer = operator
                    .get_snapshot(request)
                    .await
                    .expect("the rows")
                    .into_inner();
                micros.push(began.elapsed().as_micros());
                told = (
                    answer.workers.len(),
                    answer.truncated,
                    answer.encoded_len(),
                );
            }
            micros.sort_unstable();
            writeln!(
                out,
                "OPERATORS_SNAPSHOT workers={workers} limit={limit} models={asked} rows={} truncated={} bytes={} median_us={} store_cut_median_us={}",
                told.0, told.1, told.2, micros[5], cuts[5]
            )
            .expect("stdout");
        }
        // A watch is open: the feed keeps the cut its start took, and each
        // snapshot shares it.
        let mut gateway = client(leader.gateway).await;
        let mut watching = gateway
            .watch_workers(stated(proto::WatchWorkersRequest::default()))
            .await
            .expect("opened")
            .into_inner();
        watching
            .message()
            .await
            .expect("the stream is open")
            .expect("a page");
        let mut micros = Vec::new();
        for _ in 0..11 {
            let request = stated(proto::SnapshotRequest {
                slices: vec![proto::snapshot_request::Slice::Workers.into()],
                limit: 1,
                ..proto::SnapshotRequest::default()
            });
            let began = Instant::now();
            let answer = operator
                .get_snapshot(request)
                .await
                .expect("the rows")
                .into_inner();
            micros.push(began.elapsed().as_micros());
            assert_eq!(answer.workers.len(), 1);
        }
        micros.sort_unstable();
        writeln!(
            out,
            "OPERATORS_SNAPSHOT workers={workers} limit=1 watched=1 median_us={} most_us={}",
            micros[5], micros[10]
        )
        .expect("stdout");
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "a measurement: run by name"]
async fn the_operators_snapshots_numbers_at_1000_workers() {
    operators_snapshot_numbers(1_000).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "a measurement: run by name"]
async fn the_operators_snapshots_numbers_at_10000_workers() {
    operators_snapshot_numbers(10_000).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "a measurement: run by name"]
async fn the_snapshots_numbers_at_512_workers() {
    snapshots_numbers(512).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "a measurement: run by name"]
async fn the_snapshots_numbers_at_4096_workers() {
    snapshots_numbers(4_096).await;
}

/// The principal is the transport's, never a field a peer fills in: a hello
/// that names another replica is refused, and so is a call of the scope
/// that states no name.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_standby_is_the_principal_its_call_states() {
    let dir = tempfile::tempdir().expect("a temp dir");
    with_leader(shared(&dir), |leader| async move {
        writes(&leader.store, 1);
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
        writes(&leader.store, 1);
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
        writes(&leader.store, 1);
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
        writes(&leader.store, 3);
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

fn count(n: usize) -> u64 {
    u64::try_from(n).expect("a count")
}

/// The process's CPU time so far, user and system, in milliseconds: the
/// kernel's two counts, in ticks of 10 ms.
fn cpu_ms() -> u64 {
    let stat = std::fs::read_to_string("/proc/self/stat").expect("the process's stat");
    let fields = stat.rsplit_once(')').expect("the command's name").1;
    let ticks = |nth: usize| {
        fields
            .split_whitespace()
            .nth(nth)
            .and_then(|field| field.parse::<u64>().ok())
            .expect("a count of ticks")
    };
    (ticks(11) + ticks(12)) * 10
}

/// The process's resident set, in kilobytes.
fn resident_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .expect("the process's status")
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(|rest| rest.split_whitespace().next()?.parse().ok())
        .expect("the resident set")
}

/// One replica of a pair over one lease, as a case reaches it.
struct Replica {
    replication: SocketAddr,
    admin: SocketAddr,
    store: Arc<MemoryStore>,
    readiness: Readiness,
    stop: oneshot::Sender<()>,
    served: JoinSet<()>,
}

/// The leader's address and the standby's: one host, two addresses, so
/// both listen on one replication port as replicas configured alike do.
/// Linux routes the second with no alias. Another host refuses its bind:
/// a case that binds it is ignored there, with that reason.
const LEADER: IpAddr = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
const FOLLOWER: IpAddr = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2));

impl Replica {
    /// Binds a replica of the lease at `place` on `ip`, its replication
    /// listener at `port`, and serves it. `0` takes a number of the pairs'
    /// own ([`pair::ports`]), the next one free on `ip` and, for a pair's
    /// leader, on its standby's address too ([`pair::free`]): the number a
    /// standby of this replica then binds on its own address.
    async fn serve(id: &str, ip: IpAddr, port: u16, place: LeasePlace) -> Self {
        let listener = |name: &str, scopes: &[Scope], port| Listener {
            name: name.to_owned(),
            addr: SocketAddr::new(ip, port),
            scopes: scopes.to_vec(),
            tls: None,
        };
        let config = |port| Config {
            controller_id: id.to_owned(),
            listeners: vec![
                listener("gateway", &[Scope::Gateway, Scope::Ingestion], 0),
                listener("control", &[Scope::Control], 0),
                listener("replication", &[Scope::Replication], port),
            ],
            admin: SocketAddr::new(ip, 0),
            shutdown_settle: Duration::from_millis(50),
            shutdown_grace: Duration::from_secs(5),
            load_poll_interval: Config::DEFAULT_LOAD_POLL_INTERVAL,
            decide_shards: 2,
            discovery: None,
            election: ElectionConfig {
                lease: place.clone(),
                settings: settings(),
                advertise: None,
            },
        };
        let mut controller = if port == 0 {
            bind_drawn(ip, (ip == LEADER).then_some(FOLLOWER), config).await
        } else {
            Controller::bind(config(port)).await.expect("bind")
        };
        for name in ["control", "replication"] {
            controller
                .identity_as_told(name)
                .expect("a listener of the configuration");
        }
        let replication = controller
            .addresses()
            .of(Scope::Replication)
            .expect("the scope is mounted");
        let admin = controller.addresses().admin;
        let (store, readiness) = (controller.store(), controller.readiness());
        let (stop, stopped) = oneshot::channel::<()>();
        let mut served = JoinSet::new();
        served.spawn(async move {
            let shutdown = async {
                let _ = stopped.await;
            };
            controller.serve(shutdown).await.expect("a clean drain");
        });
        Self {
            replication,
            admin,
            store,
            readiness,
            stop,
            served,
        }
    }

    /// Resolves once the replica leads, within 30 s.
    async fn leads(&mut self) {
        let led = self.readiness.wait_for(State::Leader);
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(30), led).await,
            Ok(Some(State::Leader))
        );
    }

    /// Resolves once the replica's state is `state`, within 30 s.
    async fn reads(&mut self, state: State) {
        let read = self.readiness.wait_for(state);
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(30), read).await,
            Ok(Some(state))
        );
    }

    /// The planned stop, which releases the lease, and its drain.
    async fn stop(mut self) {
        self.stop.send(()).expect("the serve still runs");
        let drained = tokio::time::timeout(Duration::from_secs(30), self.served.join_next()).await;
        assert!(
            matches!(drained, Ok(Some(Ok(())))),
            "a clean drain within 30 s"
        );
    }

    /// What the replica holds, without its instants: its place, then its
    /// epoch and its rows.
    fn held(&self) -> ((u64, u64), impl PartialEq + std::fmt::Debug) {
        let cut = self.store.checkpoint();
        let rows: Vec<_> = cut
            .rows
            .into_iter()
            .map(|row| (row.id, row.facts, row.status, row.revision))
            .collect();
        ((cut.at.seq, cut.at.term), (cut.epoch, rows))
    }

    /// How many load rows the replica's own table holds.
    fn load_rows(&self) -> usize {
        let rows = self.store.checkpoint().rows;
        rows.iter().map(|row| self.store.ranks(row.id)).sum()
    }

    /// Waits, within 20 s, until the replica stands at `place`.
    async fn stands_at(&self, place: (u64, u64)) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while self.held().0 != place {
            assert!(
                Instant::now() < deadline,
                "the replica stands at {:?}, not at {place:?}, after 20 s",
                self.held().0
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

/// How many polls the replica whose admin listener is `admin` has sent, by
/// its metrics page.
async fn polls(admin: SocketAddr) -> u64 {
    let (status, page) = http_get(admin, "/metrics").await.expect("the metrics page");
    assert_eq!(status, 200);
    let counter = "smg_policy_controller_load_polls_total{mode=\"poll\"} ";
    let count = page.lines().find_map(|line| line.strip_prefix(counter));
    count
        .and_then(|count| count.trim().parse().ok())
        .expect("the polls' counter")
}

/// The `Leader` call's answer at `addr`.
async fn leader_info(addr: SocketAddr) -> proto::LeaderInfo {
    client(addr)
        .await
        .leader(proto::LeaderRequest::default())
        .await
        .expect("every listener answers")
        .into_inner()
}

/// The snapshot of a replica that took the lease, within 20 s: between its
/// take and the take's own write the call is `UNAVAILABLE`.
async fn promoted(addr: SocketAddr) -> proto::Snapshot {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        match snapshot(addr).await {
            Ok(served) => return served,
            Err(wait) => {
                assert_eq!(wait.code(), Code::Unavailable, "no snapshot: {wait}");
                assert!(Instant::now() < deadline, "no snapshot in 20 s: {wait}");
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "the follower binds 127.0.0.2: off Linux that takes a loopback alias"
)]
async fn a_promoted_standby_holds_every_row_of_its_leader_and_no_load_row() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut leader = Replica::serve("replication-0", LEADER, 0, shared(&dir)).await;
    leader.leads().await;
    let began = leader.held().0;
    let port = leader.replication.port();
    let mut standby = Replica::serve("replication-1", FOLLOWER, port, shared(&dir)).await;
    // A ready worker the load monitor polls, at an address nobody answers.
    let polled = WorkerFacts {
        wire_modes: WireModes::GRPC,
        ..facts("grpc://127.0.0.1:1", 1, 0)
    };
    let polled = leader.store.upsert(polled).id();
    let revision = leader.store.get(polled).expect("the row").revision;
    leader
        .store
        .set_status(polled, WorkerStatus::Ready, revision);
    // A row nobody polls, with a load row on the leader's table.
    let quiet = writes(&leader.store, 0);
    leader
        .store
        .record(quiet, 0, running(1), Instant::now(), LoadSource::Stream);
    // The leader writes while the standby follows.
    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let (store, stop) = (Arc::clone(&leader.store), Arc::clone(&stop));
        thread::spawn(move || churn(&store, &stop))
    };
    // The events the case counts are waited for, never timed: a load row
    // of the writer is no event of the log.
    written_past(&leader.store, began.0 + 100).await;
    stop.store(true, Ordering::Relaxed);
    let rounds = writer.join().expect("the writer");
    let (head, wrote) = leader.held();
    let events = head.0 - began.0;
    assert!(events > 100, "{rounds} rounds wrote {events} events");
    standby.stands_at(head).await;
    assert_eq!(standby.held().1, wrote, "{rounds} rounds");
    // No load row crossed: the leader's table holds the quiet row's, and
    // the standby's holds none.
    assert!(leader.store.load(quiet, 0).is_some());
    assert_eq!(standby.load_rows(), 0, "a standby is sent no load row");
    // The leader's load monitor polls the ready worker. The standby holds
    // the same row and polls nobody while it follows: a while later, its
    // counter still reads none.
    let deadline = Instant::now() + Duration::from_secs(20);
    while polls(leader.admin).await == 0 {
        assert!(Instant::now() < deadline, "the leader polls its worker");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    tokio::time::sleep(Duration::from_millis(1_200)).await;
    assert_eq!(polls(standby.admin).await, 0, "a follower polls nobody");
    // The standby says how far behind it is; the leader follows nobody.
    let info = leader_info(standby.replication).await;
    assert!(!info.is_leader && info.synced, "{info:?}");
    let lag = info.sync.map(|sync| sync.replication_lag_ms);
    assert!(lag.is_some_and(|lag| lag < 5_000), "{lag:?}");
    let own = leader_info(leader.replication).await;
    assert!(
        own.is_leader && !own.synced && own.sync.is_none(),
        "{own:?}"
    );

    // The leader stops: the standby takes the lease with what it held and
    // serves it under a term of its own.
    leader.stop().await;
    standby.leads().await;
    let served = promoted(standby.replication).await;
    assert_eq!(
        served.replication_seq,
        head.0 + 1,
        "its term starts after the last event"
    );
    assert!(served.replication_term > head.1);
    let (place, holds) = standby.held();
    assert_eq!(place, (served.replication_seq, served.replication_term));
    assert_eq!(holds, wrote);
    // It reads its engines itself from its take: its own load monitor
    // polls the ready worker.
    let deadline = Instant::now() + Duration::from_secs(20);
    while polls(standby.admin).await == 0 {
        assert!(
            Instant::now() < deadline,
            "the promoted standby polls its worker"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // A leader follows nobody: its answer says no sync state.
    let promoted = leader_info(standby.replication).await;
    assert!(
        promoted.is_leader && !promoted.synced && promoted.sync.is_none(),
        "{promoted:?}"
    );
    standby.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "the follower binds 127.0.0.2: off Linux that takes a loopback alias"
)]
async fn a_standby_that_joins_late_is_restored_whole_from_a_snapshot_past_one_message() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut leader = Replica::serve("replication-0", LEADER, 0, shared(&dir)).await;
    leader.leads().await;
    let began = leader.held().0;
    // More status changes than the log retains of the watched kinds
    // (4,096): the place a new standby names is gone from it.
    let id = leader.store.upsert(facts("http://late:8000", 1, 0)).id();
    let revision = leader.store.get(id).expect("the row").revision;
    for round in 1..=4_300 {
        let status = if round % 2 == 1 {
            WorkerStatus::Ready
        } else {
            WorkerStatus::NotReady
        };
        leader
            .store
            .set_status(id, status, revision)
            .expect("the row's revision");
    }
    // Rows of a few kilobytes each, by their key: more than one message of
    // the transport holds, so the snapshot is several parts.
    let wide = "k".repeat(2_000);
    for row in 0..1_500 {
        leader
            .store
            .upsert(facts(&format!("http://{wide}-{row}:8000"), 1, 0));
    }
    writes(&leader.store, 3);
    let port = leader.replication.port();
    let standby = Replica::serve("replication-1", FOLLOWER, port, shared(&dir)).await;
    let (head, wrote) = leader.held();
    assert!(head.0 - began.0 > 4_300);
    standby.stands_at(head).await;
    assert_eq!(standby.held().1, wrote);
    // It follows the log from the snapshot's place.
    leader.store.upsert(facts("http://after:8000", 1, 0));
    let (head, wrote) = leader.held();
    standby.stands_at(head).await;
    assert_eq!(standby.held().1, wrote);
    standby.stop().await;
    leader.stop().await;
}

/// A promoted standby holds its leader's history from the first write: a
/// reader that holds nothing is fed it whole, then the new term's start. A
/// standby restored from a snapshot tells such a reader to start again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "the follower binds 127.0.0.2: off Linux that takes a loopback alias"
)]
async fn a_promoted_standby_feeds_a_reader_that_holds_nothing_from_its_leaders_first_write() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut leader = Replica::serve("replication-0", LEADER, 0, shared(&dir)).await;
    leader.leads().await;
    let port = leader.replication.port();
    let mut standby = Replica::serve("replication-1", FOLLOWER, port, shared(&dir)).await;
    writes(&leader.store, 3);
    let (head, wrote) = leader.held();
    standby.stands_at(head).await;
    assert_eq!(standby.held().1, wrote);
    leader.stop().await;
    standby.leads().await;
    let served = promoted(standby.replication).await;
    let mut reader = replicate(standby.replication, place(0, 0))
        .await
        .expect("a stream");
    let fed = reader.until(served.replication_seq).await;
    let mut twin = Twin::default();
    for event in &fed {
        twin.apply(event).expect("an event the leader journals");
    }
    let apart = twin.apart(&Twin::of(&served));
    assert!(apart.is_empty(), "{apart:?}");
    // The first leader's term's start and its writes, then the promoted
    // replica's own term's start.
    let starts: Vec<_> = fed
        .iter()
        .enumerate()
        .filter(|(_, event)| matches!(event.event, Some(Sent::TermStart(_))))
        .map(|(index, _)| index)
        .collect();
    assert_eq!(starts, [0, fed.len() - 1], "{} events", fed.len());
    standby.stop().await;
}

/// The listener's overall word: the health service's answer for no name.
async fn overall(addr: SocketAddr) -> ServingStatus {
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("a uri")
        .connect()
        .await
        .expect("the listener accepts");
    HealthClient::new(channel)
        .check(HealthCheckRequest {
            service: String::new(),
        })
        .await
        .expect("the health service answers for no name")
        .into_inner()
        .status()
}

/// The sample of `series`, a series of no label, on the replica's metrics
/// page.
async fn sample(admin: SocketAddr, series: &str) -> String {
    let (status, page) = http_get(admin, "/metrics").await.expect("the metrics page");
    assert_eq!(status, 200);
    let sample = page
        .lines()
        .find_map(|line| line.strip_prefix(series)?.strip_prefix(' '));
    sample.expect("the series").to_owned()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "the follower binds 127.0.0.2: off Linux that takes a loopback alias"
)]
async fn a_standby_passes_its_probes_only_while_it_is_synced_with_its_leader() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut leader = Replica::serve("replication-0", LEADER, 0, shared(&dir)).await;
    leader.leads().await;
    let port = leader.replication.port();

    // A standby that cannot reach its leader: it dials the leader's
    // address at its own replication port, where no replication listener
    // answers. It serves, and no probe of it passes.
    let mut apart = Replica::serve("replication-1", FOLLOWER, 0, shared(&dir)).await;
    assert_ne!(apart.replication.port(), port);
    apart.reads(State::Syncing).await;
    let (status, body) = http_get(apart.admin, "/readiness")
        .await
        .expect("the probe");
    assert_eq!(
        (status, body.as_str()),
        (503, r#"{"status":"not ready","reason":"syncing"}"#)
    );
    assert_eq!(overall(apart.replication).await, ServingStatus::NotServing);
    let state = "smg_policy_controller_state{state=\"syncing\"}";
    assert_eq!(sample(apart.admin, state).await, "1");
    assert_eq!(
        sample(apart.admin, "smg_policy_controller_replication_synced").await,
        "0"
    );
    apart.stop().await;

    // One that follows its leader passes them once its first frame is
    // applied, and says so on its metrics page.
    let mut standby = Replica::serve("replication-1", FOLLOWER, port, shared(&dir)).await;
    standby.reads(State::Follower).await;
    let (status, body) = http_get(standby.admin, "/readiness")
        .await
        .expect("the probe");
    assert_eq!(
        (status, body.as_str()),
        (200, r#"{"status":"ready","role":"follower"}"#)
    );
    // The word is told after the state turns.
    let deadline = Instant::now() + Duration::from_secs(20);
    while overall(standby.replication).await != ServingStatus::Serving {
        assert!(
            Instant::now() < deadline,
            "a synced standby's word is not SERVING after 20 s"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        sample(standby.admin, "smg_policy_controller_replication_synced").await,
        "1"
    );
    assert_eq!(sample(standby.admin, state).await, "0");
    standby.stop().await;
    leader.stop().await;
}
