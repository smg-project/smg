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

use std::{
    collections::BTreeMap,
    future::Future,
    io::Write as _,
    net::{IpAddr, Ipv4Addr, SocketAddr},
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
    controller::{Config, Controller, Election, LeasePlace, Listener, Readiness, Scope, State},
    elector::{ElectionSettings, Elector, FileLease, Tick},
    proto::{
        self, policy_controller_client::PolicyControllerClient, replica_frame::Frame,
        replication_event::Event as Sent, worker_event::Event,
    },
    store::{Kinds, LoadSink, LoadSource, MemoryStore, WorkerFacts, WorkerStatus, WorkersStore},
    LoadRow, WireModes, WorkerId,
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

/// Serves a controller over a lease at `place` in the case's process, and
/// runs `case` against it once it leads.
async fn with_leader<Run>(place: LeasePlace, case: impl FnOnce(Leader) -> Run)
where
    Run: Future<Output = ()>,
{
    let mut controller = Controller::bind(Config {
        controller_id: "replication-0".to_owned(),
        listeners: vec![
            listener("gateway", &[Scope::Gateway, Scope::Ingestion]),
            listener("control", &[Scope::Control]),
            listener("replication", &[Scope::Replication]),
        ],
        admin: loopback(),
        shutdown_settle: Duration::from_millis(50),
        shutdown_grace: Duration::from_secs(5),
        load_poll_interval: Config::DEFAULT_LOAD_POLL_INTERVAL,
        decide_shards: 2,
        election: Election {
            lease: place,
            settings: settings(),
            advertise: None,
        },
    })
    .await
    .expect("bind");
    // The control and the replication listeners take an identity as told.
    for name in ["control", "replication"] {
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
/// shared lease, so the store journals its load rows, and one the case's
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
        assert!(matches!(took, Ok(Tick::Acquired { .. })), "{took:?}");
        // A release inside the take's wait hands the wait on, so the holder
        // before leads first: its take opens no fence until a renewal one
        // renew deadline after it.
        let led = Instant::now() + settings().renew_deadline;
        let renewed = before.tick(led);
        assert!(matches!(renewed, Ok(Tick::Renewed { .. })), "{renewed:?}");
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
    /// batch that is on its way, never through five silent seconds).
    async fn batch(&mut self) -> proto::ReplicationBatch {
        let mut found = None;
        for _ in 0..5 {
            let batch = self.frame().await;
            if batch.resync_required || !batch.events.is_empty() {
                found = Some(batch);
                break;
            }
        }
        found.expect("a batch within five heartbeats")
    }

    /// The next heartbeat, past at most 64 batches.
    async fn beat(&mut self) -> proto::ReplicationBatch {
        let mut found = None;
        for _ in 0..64 {
            let batch = self.frame().await;
            let part = batch.snapshot.is_some();
            if !batch.resync_required && batch.events.is_empty() && !part {
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
    /// together: the rows, the load rows, the place every part names and
    /// the epoch. The stream stands after it: the log follows from that
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

fn facts(key: &str, incarnation: u64, max_running: u32) -> WorkerFacts {
    WorkerFacts {
        key: key.to_owned(),
        engine: key.to_owned(),
        incarnation,
        models: vec!["model-a".to_owned()],
        dp_size: 1,
        max_running: (max_running != 0).then_some(max_running),
        ..WorkerFacts::default()
    }
}

fn running(count: u32) -> LoadRow {
    LoadRow {
        running: count,
        ..LoadRow::default()
    }
}

/// One worker and `loads` load rows of it.
fn writes(store: &MemoryStore, loads: u32) -> WorkerId {
    let id = store.upsert(facts("http://a:8000", 1, 0)).id();
    let now = Instant::now();
    for count in 0..loads {
        store.record(id, 0, running(count), now, LoadSource::Stream);
    }
    id
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
        // and three load rows stand after that place.
        let start = base.replication_seq - 5;
        assert_ne!(start, 0, "a run draws where its places start");
        let nth = |write: u64| start + write;

        // A standby with nothing: the take, the registration, the rows.
        let mut fresh = replicate(leader.replication, place(0, 0))
            .await
            .expect("a stream");
        let first = fresh.batch().await;
        assert_eq!((first.first_seq, first.term), (nth(1), term));
        assert_eq!(
            cases(&first.events),
            ["term", "membership", "load", "load", "load"]
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
        let counted = LoadRow {
            resident_tokens: 4_000,
            resident_counted: true,
            ..running(9)
        };
        leader
            .store
            .record(id, 0, counted, Instant::now(), LoadSource::Stream);
        let next = joined.batch().await;
        assert_eq!((next.first_seq, next.term), (nth(6), term));
        assert_eq!(cases(&next.events), ["load"]);
        let Some(Sent::Load(told)) = &next.events[0].event else {
            panic!("the load row");
        };
        assert!(told.resident_counted, "the flag crosses beside the row");
        // The first stream reads the same event: one log, two readers.
        assert_eq!(
            ageless_events(&fresh.batch().await.events),
            ageless_events(&next.events)
        );
        // An idle second: the leader says where its head stands.
        let beat = joined.beat().await;
        assert_eq!((beat.first_seq, beat.term), (nth(7), term));

        // A position past the head and a position that names no term: each
        // is told to re-sync and restored on the stream it holds, at the
        // leader's head, and reads on from there.
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
        // Rounds of 100 load rows, each read whole before the next is
        // written. A standby that acknowledges each round is never 300
        // behind, were its acknowledgement a whole round late.
        let mut steady = replicate(leader.replication, hello.clone())
            .await
            .expect("a stream");
        let base = snapshot(leader.replication).await.expect("a snapshot");
        let mut head = base.replication_seq;
        for _ in 0..8 {
            for count in 0..100 {
                leader
                    .store
                    .record(id, 0, running(count), Instant::now(), LoadSource::Stream);
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

/// What a standby holds of a snapshot and the events after it: the rows and
/// the load rows, under the store's own rules (a load row of no member is
/// dropped; a row that leaves takes its load rows; a status write moves
/// no revision).
#[derive(Debug, Default, PartialEq)]
struct Twin {
    rows: BTreeMap<u64, proto::WorkerRow>,
    loads: BTreeMap<(u64, u32), proto::LoadRow>,
    /// The membership epoch: the snapshot's, then each membership event's.
    epoch: u64,
}

/// A load row without its age: the age is the send's.
fn ageless(row: Option<&proto::LoadRow>) -> proto::LoadRow {
    proto::LoadRow {
        report_age_ms: 0,
        ..row.copied().expect("a load row")
    }
}

/// Events with every load row's age taken out: two sends of one event
/// differ by it.
fn ageless_events(events: &[proto::ReplicationEvent]) -> Vec<proto::ReplicationEvent> {
    let mut events = events.to_vec();
    for event in &mut events {
        if let Some(Sent::Load(proto::LoadSample {
            load: Some(row), ..
        })) = event.event.as_mut()
        {
            row.report_age_ms = 0;
        }
    }
    events
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
        Self {
            // A row's admission age is each send's own, as a load row's is.
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
            loads: snapshot
                .loads
                .iter()
                .map(|sample| {
                    (
                        (sample.worker_id, sample.dp_rank),
                        ageless(sample.load.as_ref()),
                    )
                })
                .collect(),
            epoch: snapshot.membership_epoch,
        }
    }

    fn leave(&mut self, id: u64) {
        self.rows.remove(&id);
        self.loads.retain(|(of, _), _| *of != id);
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
        for (key, row) in &self.loads {
            match other.loads.get(key) {
                None => apart.push(format!("load {key:?} only here")),
                Some(theirs) if theirs != row => {
                    apart.push(format!("load {key:?}: here {row:?}, there {theirs:?}"));
                }
                Some(_) => {}
            }
        }
        for key in other.loads.keys() {
            if !self.loads.contains_key(key) {
                apart.push(format!("load {key:?} only there"));
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
                    self.leave(replaced.old.as_ref().expect("the old row").id);
                    let new = replaced.new.clone().expect("the new row");
                    self.rows.insert(new.id, new);
                }
                Event::Removed(removed) => self.leave(removed.worker_id),
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
            Sent::Load(sample) => {
                if self.rows.contains_key(&sample.worker_id) {
                    self.loads.insert(
                        (sample.worker_id, sample.dp_rank),
                        ageless(sample.load.as_ref()),
                    );
                }
            }
            Sent::TermStart(_) => {}
            other => return Err(format!("a kind the leader does not journal: {other:?}")),
        }
        Ok(())
    }
}

/// Registers, flips, refreshes, replaces and removes workers over a few
/// keys and writes their load rows, until `stop`.
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
    while store.cut(Kinds::All).at.seq <= seq {
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
        // A standby's snapshot rides its stream: the row and its load row.
        let restored = snapshot(leader.replication).await.expect("a snapshot");
        let rows: Vec<u64> = restored.workers.iter().map(|row| row.id).collect();
        assert_eq!(rows, [id.0]);
        assert_eq!(restored.loads.len(), 1);
        // The call is the operators' read: the replication listener and the
        // gateways' mount no scope of it, and the control scope does not
        // serve it yet.
        for (addr, code) in [
            (leader.replication, Code::PermissionDenied),
            (leader.gateway, Code::PermissionDenied),
            (leader.control, Code::Unimplemented),
        ] {
            let refused = get_snapshot(addr).await.expect_err("no snapshot");
            assert_eq!(refused.code(), code, "{addr}");
        }
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
        // Every row and load row the leader holds, at its head.
        let held = leader.store.cut(Kinds::All);
        let at = (restored.replication_seq, restored.replication_term);
        assert_eq!(at, (held.at.seq, held.at.term));
        assert_eq!(restored.membership_epoch, held.epoch);
        let told: Vec<u64> = restored.workers.iter().map(|row| row.id).collect();
        let rows: Vec<u64> = held.rows.iter().map(|row| row.id.0).collect();
        assert_eq!(told, rows);
        assert_eq!(restored.loads.len(), held.loads.len());
        // The log follows from the snapshot's place, on the same stream.
        leader
            .store
            .record(id, 0, running(7), Instant::now(), LoadSource::Stream);
        let next = standby.batch().await;
        assert_eq!((next.first_seq, next.term), (at.0 + 1, at.1));
        assert_eq!(cases(&next.events), ["load"]);
    })
    .await;
}

/// A measurement, run by name with `--ignored --nocapture`: a standby's
/// snapshot at the design's top fleet and at 4,096 workers of a production
/// row's shape (a 56-byte address, one 36-byte model, four labels, a zone)
/// with 0, 1, 8 and 64 load rows a worker. Each line: the parts, their
/// bytes together and the widest one, and how long a standby that asks
/// waits for the whole (the leader's cut, its parts, the client's read),
/// with the CPU of both ends over the eleven snapshots timed.
/// Then the leader's memory for a held cut, over the rows alone (what a
/// watch's kept cut holds) and with 64 load rows a worker: the resident
/// growth of 32 streams left inside their snapshots at 32 places. Each
/// fleet is measured in a process of its own: a resident reading starts
/// where no cut of another fleet was let go.
async fn snapshots_numbers(workers: u32) {
    // A load row with every count set, as a busy engine reports it.
    let busy = LoadRow {
        running: 48,
        queued: 12,
        waiting_uncached_tokens: 180_000,
        kv_usage_permille: 870,
        active_kv_permille: 640,
        live_rate: 3_200,
        resident_tokens: 1_500_000,
        resident_counted: true,
        report_age_ms: 40,
        max_running: 64,
        ..LoadRow::default()
    };
    let dir = tempfile::tempdir().expect("a temp dir");
    with_leader(shared(&dir), |leader| async move {
        let mut ids = Vec::new();
        for worker in 0..workers {
            let address =
                format!("http://engine-{worker:04}.serving-pool.example.internal:8000");
            let labels = (0..4)
                .map(|n| (format!("label-key-{n:02}"), format!("value-{worker:04}")))
                .collect();
            let wide = WorkerFacts {
                models: vec!["organization/model-name-70b-instruct".to_owned()],
                labels,
                zone: Some("region-1-zone-a".to_owned()),
                engine_kind: "engine-kind".to_owned(),
                ..facts(&address, 1, 64)
            };
            ids.push(leader.store.upsert(wide).id());
        }
        let now = Instant::now();
        let mut out = std::io::stdout();
        let mut written = 0u16;
        let mut held = Vec::new();
        for each in [0u16, 1, 8, 64] {
            for rank in written..each {
                for id in &ids {
                    leader
                        .store
                        .record(*id, rank, busy, now, LoadSource::Stream);
                }
            }
            written = each;
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
                "SNAPSHOT workers={workers} load_rows_each={each} parts={} bytes={} widest_part={} median_us={} cpu_ms_of_11={cpu}",
                parts.len(),
                parts.iter().sum::<usize>(),
                parts.iter().max().copied().unwrap_or(0),
                micros[5]
            )
            .expect("stdout");
            if each != 0 && each != 64 {
                continue;
            }
            // Streams left inside their snapshots, each at a place of its
            // own: a load row between two moves the head, so no two share a
            // cut. Each read the word and its first part, and no further.
            // Over the rows alone a cut is what a watch's kept cut holds.
            // The first 32 stay held: the second reading reuses nothing
            // they let go.
            let before = resident_kb();
            for turn in 0..32u32 {
                let id = ids[turn as usize % ids.len()];
                leader.store.record(id, 0, busy, now, LoadSource::Stream);
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
                "HELD workers={workers} load_rows_each={each} cuts=32 resident_kb_a_cut={}",
                grew / 32
            )
            .expect("stdout");
        }
        assert_eq!(held.len(), 64, "each stream still holds its cut");
    })
    .await;
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

/// The fence at the wire. A leader whose lease is gone ends an open stream
/// `ABORTED` at its next look of the lease. A hello it still waited for is
/// refused the same way, though it names a place the log does not hold: a
/// replica that lost the lead tells no standby to re-sync. Both calls are
/// refused after it.
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
        // A second call says no hello yet, from a task of its own. Nothing
        // orders it ahead of the call made here: it has that round
        // trip, and the leader's wait for its next look at the lease, to
        // be admitted while the leader leads.
        let mut waits = client(leader.replication).await;
        let (late, sent) = mpsc::channel(1);
        let mut waiting = JoinSet::new();
        waiting.spawn({
            let mut waits = waits.clone();
            async move {
                waits
                    .replicate(stated(ReceiverStream::new(sent)))
                    .await
                    .map(|_| ())
            }
        });
        waits
            .leader(proto::LeaderRequest::default())
            .await
            .expect("the leader's word while it leads");
        std::fs::remove_file(&lease).expect("the lease file");
        assert_eq!(open.ended().await, Some(Code::Aborted));
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
        assert_eq!(refused.code(), Code::Aborted);
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

/// Writes `rate` load rows a second over `ids` for `window`, in steps of
/// 10 ms, and says how many it wrote.
fn paced(store: &MemoryStore, ids: &[WorkerId], rate: u64, window: Duration) -> u64 {
    let start = Instant::now();
    let mut written = 0u64;
    while start.elapsed() < window {
        let elapsed = u64::try_from(start.elapsed().as_millis()).expect("a short window");
        while written < elapsed * rate / 1_000 {
            let index = usize::try_from(written).expect("a count") % ids.len();
            let row = running(u32::try_from(written % 12).expect("under twelve"));
            store.record(ids[index], 0, row, Instant::now(), LoadSource::Stream);
            written += 1;
        }
        thread::sleep(Duration::from_millis(10));
    }
    written
}

/// What one window read: the rows written, the events and the bytes of the
/// frames its standbys took together, and the process's CPU over it.
struct Window {
    written: u64,
    events: u64,
    bytes: u64,
    cpu_ms: u64,
    millis: u64,
}

/// Four seconds of `rate` load rows a second with so many standbys reading
/// and acknowledging, each from the head it joined at.
async fn window(leader: &Leader, ids: &Arc<Vec<WorkerId>>, rate: u64, standbys: usize) -> Window {
    let stop = Arc::new(AtomicBool::new(false));
    let mut readers = JoinSet::new();
    for _ in 0..standbys {
        let head = snapshot(leader.replication).await.expect("a snapshot");
        let mut standby = replicate(
            leader.replication,
            place(head.replication_seq, head.replication_term),
        )
        .await
        .expect("a stream");
        let stop = Arc::clone(&stop);
        readers.spawn(async move {
            let (mut events, mut bytes) = (0u64, 0u64);
            while !stop.load(Ordering::Relaxed) {
                // A batch or a heartbeat arrives inside a second and a half.
                let next =
                    tokio::time::timeout(Duration::from_millis(1_500), standby.batches.message());
                let Ok(Ok(Some(batch))) = next.await else {
                    break;
                };
                bytes += count(batch.encoded_len());
                if !batch.events.is_empty() {
                    events += count(batch.events.len());
                    let last = batch.first_seq + count(batch.events.len()) - 1;
                    if standby.frames.send(ack(last)).await.is_err() {
                        break;
                    }
                }
            }
            (events, bytes)
        });
    }
    let before = cpu_ms();
    let started = Instant::now();
    let writer = {
        let (store, ids) = (Arc::clone(&leader.store), Arc::clone(ids));
        thread::spawn(move || paced(&store, &ids, rate, Duration::from_secs(4)))
    };
    while !writer.is_finished() {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let written = writer.join().expect("the writer");
    // The last batches reach the standbys.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let spent = cpu_ms() - before;
    let millis = u64::try_from(started.elapsed().as_millis()).expect("a short window");
    stop.store(true, Ordering::Relaxed);
    let (mut events, mut bytes) = (0, 0);
    while let Some(read) = readers.join_next().await {
        let (read_events, read_bytes) = read.expect("a reader");
        events += read_events;
        bytes += read_bytes;
    }
    Window {
        written,
        events,
        bytes,
        cpu_ms: spent,
        millis,
    }
}

/// The Numbers table's rows of the stream, run by name with `--ignored
/// --nocapture`: the resident memory of a full log; the bytes and the CPU
/// at the design's busy rate (ten load rows a rank a second) with no
/// standby and with one; the snapshot's bytes and time; the idle
/// stream. Leader and standby share the process, so the CPU is both ends'.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "a measurement: run by name"]
async fn the_streams_numbers() {
    let dir = tempfile::tempdir().expect("a temp dir");
    with_leader(shared(&dir), |leader| async move {
        let mut lines = Vec::new();
        let mut ids = Vec::new();
        for ranks in [64usize, 512] {
            while ids.len() < ranks {
                let key = format!("http://w{}:8000", ids.len());
                ids.push(leader.store.upsert(facts(&key, 1, 0)).id());
            }
            if ranks == 64 {
                // A full log: the load rows' ring holds 65,536.
                let before = resident_kb();
                let now = Instant::now();
                for turn in 0..70_000usize {
                    let id = ids[turn % ids.len()];
                    leader
                        .store
                        .record(id, 0, running(3), now, LoadSource::Stream);
                }
                lines.push(format!(
                    "RESIDENT before_kb={before} after_70000_load_rows_kb={}",
                    resident_kb()
                ));
                // A catch-up: 60,000 load rows that stand in the log, read by
                // one standby from the place before them.
                let from = snapshot(leader.replication).await.expect("a snapshot");
                for turn in 0..60_000usize {
                    let id = ids[turn % ids.len()];
                    leader
                        .store
                        .record(id, 0, running(5), now, LoadSource::Stream);
                }
                let (cpu, at) = (cpu_ms(), Instant::now());
                let mut behind = replicate(
                    leader.replication,
                    place(from.replication_seq, from.replication_term),
                )
                .await
                .expect("a stream");
                let events = behind.until(from.replication_seq + 60_000).await.len();
                lines.push(format!(
                    "CATCHUP events={events} micros={} cpu_ms={}",
                    at.elapsed().as_micros(),
                    cpu_ms() - cpu
                ));
            }
            let fleet = Arc::new(ids.clone());
            let rate = count(ranks) * 10;
            for standbys in [0usize, 1] {
                let read = window(&leader, &fleet, rate, standbys).await;
                lines.push(format!(
                    "STREAM ranks={ranks} rate={rate} standbys={standbys} written={} events={} bytes={} cpu_ms={} window_ms={}",
                    read.written, read.events, read.bytes, read.cpu_ms, read.millis
                ));
            }
            let mut micros = Vec::new();
            let mut bytes = 0;
            for _ in 0..21 {
                let at = Instant::now();
                let complete = snapshot(leader.replication).await.expect("a snapshot");
                micros.push(at.elapsed().as_micros());
                bytes = complete.encoded_len();
            }
            micros.sort_unstable();
            lines.push(format!(
                "SNAPSHOT ranks={ranks} bytes={bytes} median_us={}",
                micros[10]
            ));
        }
        let fleet = Arc::new(ids.clone());
        for standbys in [0usize, 1] {
            let read = window(&leader, &fleet, 0, standbys).await;
            lines.push(format!(
                "IDLE standbys={standbys} events={} bytes={} cpu_ms={} window_ms={}",
                read.events, read.bytes, read.cpu_ms, read.millis
            ));
        }
        let mut out = std::io::stdout();
        for line in lines {
            writeln!(out, "{line}").expect("stdout");
        }
    })
    .await;
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
    /// listener at `port` (`0` takes one), and serves it.
    async fn serve(id: &str, ip: IpAddr, port: u16, place: LeasePlace) -> Self {
        let listener = |name: &str, scopes: &[Scope], port| Listener {
            name: name.to_owned(),
            addr: SocketAddr::new(ip, port),
            scopes: scopes.to_vec(),
            tls: None,
        };
        let mut controller = Controller::bind(Config {
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
            election: Election {
                lease: place,
                settings: settings(),
                advertise: None,
            },
        })
        .await
        .expect("bind");
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
    /// epoch, its rows and its load rows, each row without its age.
    fn held(&self) -> ((u64, u64), impl PartialEq + std::fmt::Debug) {
        let cut = self.store.cut(Kinds::All);
        let rows: Vec<_> = cut
            .rows
            .into_iter()
            .map(|row| (row.id, row.facts, row.status, row.revision))
            .collect();
        let loads: Vec<_> = cut
            .loads
            .into_iter()
            .map(|(id, rank, sample)| {
                let row = LoadRow {
                    report_age_ms: 0,
                    ..sample.row
                };
                (id, rank, row)
            })
            .collect();
        ((cut.at.seq, cut.at.term), (cut.epoch, rows, loads))
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
async fn a_promoted_standby_holds_every_row_and_load_row_of_its_leader() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut leader = Replica::serve("replication-0", LEADER, 0, shared(&dir)).await;
    leader.leads().await;
    let began = leader.held().0;
    let port = leader.replication.port();
    let mut standby = Replica::serve("replication-1", FOLLOWER, port, shared(&dir)).await;
    // A ready worker the load driver polls, at an address nobody answers.
    let polled = WorkerFacts {
        wire_modes: WireModes::GRPC,
        ..facts("grpc://127.0.0.1:1", 1, 0)
    };
    let polled = leader.store.upsert(polled).id();
    let revision = leader.store.get(polled).expect("the row").revision;
    leader
        .store
        .set_status(polled, WorkerStatus::Ready, revision);
    // The leader writes while the standby follows.
    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let (store, stop) = (Arc::clone(&leader.store), Arc::clone(&stop));
        thread::spawn(move || churn(&store, &stop))
    };
    tokio::time::sleep(Duration::from_millis(300)).await;
    stop.store(true, Ordering::Relaxed);
    let rounds = writer.join().expect("the writer");
    let (head, wrote) = leader.held();
    let events = head.0 - began.0;
    assert!(events > 100, "{rounds} rounds wrote {events} events");
    standby.stands_at(head).await;
    assert_eq!(standby.held().1, wrote, "{rounds} rounds");
    // The leader's driver polls the ready worker. The standby holds the
    // same row and polls nobody while it follows: a tick of its driver
    // later, its counter still reads none.
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

/// Four seconds of 5,120 load rows a second on `leader`: the rows written
/// and the process's CPU over them.
async fn busy(leader: &Replica, ids: &Arc<Vec<WorkerId>>) -> (u64, u64) {
    let before = cpu_ms();
    let writer = {
        let (store, ids) = (Arc::clone(&leader.store), Arc::clone(ids));
        thread::spawn(move || paced(&store, &ids, 5_120, Duration::from_secs(4)))
    };
    while !writer.is_finished() {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    (writer.join().expect("the writer"), cpu_ms() - before)
}

/// A measurement, run by name with `--ignored --nocapture`: a leader of 512
/// workers at the design's busy rate, alone and with a standby that follows
/// it in the same process (the CPU is both replicas' then); how long the
/// standby takes to stand at the head once the writer stops; what it says
/// of its lag; and the resident memory a following replica adds once its
/// log's ring is full.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "a measurement: run by name"]
async fn a_following_standbys_numbers() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let mut leader = Replica::serve("replication-0", LEADER, 0, shared(&dir)).await;
    leader.leads().await;
    let ids: Vec<WorkerId> = (0..512)
        .map(|n| {
            let key = format!("http://w{n}:8000");
            leader.store.upsert(facts(&key, 1, 0)).id()
        })
        .collect();
    let ids = Arc::new(ids);
    let fill = |store: &MemoryStore| {
        let now = Instant::now();
        for turn in 0..70_000usize {
            store.record(
                ids[turn % ids.len()],
                0,
                running(3),
                now,
                LoadSource::Stream,
            );
        }
    };
    let mut out = std::io::stdout();
    // The leader's own ring is full before anything is read.
    fill(&leader.store);
    let (written, cpu) = busy(&leader, &ids).await;
    writeln!(out, "BUSY standbys=0 written={written} cpu_ms={cpu}").expect("stdout");
    let alone = resident_kb();
    let port = leader.replication.port();
    let standby = Replica::serve("replication-1", FOLLOWER, port, shared(&dir)).await;
    standby.stands_at(leader.held().0).await;
    let (written, cpu) = busy(&leader, &ids).await;
    let stopped = Instant::now();
    standby.stands_at(leader.held().0).await;
    writeln!(
        out,
        "BUSY standbys=1 written={written} cpu_ms={cpu} at_the_head_after_ms={}",
        stopped.elapsed().as_millis()
    )
    .expect("stdout");
    let info = leader_info(standby.replication).await;
    writeln!(
        out,
        "LAG ms={:?} synced={}",
        info.sync.map(|sync| sync.replication_lag_ms),
        info.synced
    )
    .expect("stdout");
    // The standby's ring fills as the leader's did.
    fill(&leader.store);
    standby.stands_at(leader.held().0).await;
    writeln!(
        out,
        "RESIDENT the_leader_alone_kb={alone} with_a_standby_at_a_full_ring_kb={}",
        resident_kb()
    )
    .expect("stdout");
    standby.stop().await;
    leader.stop().await;
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
