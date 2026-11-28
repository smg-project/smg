//! The replication scope's call end to end (design 5.4), against the
//! controller's serve in the case's own process: `Replicate` from a
//! standby's place, and the snapshot a standby is restored from on it.
//!
//! No stub server: the client is the generated one and the server is the
//! controller's own serve. The writes go through the handle of its store,
//! the way the controller's own drivers write.
//!
//! One module a subject holds the cases, each a file under
//! `replication_cases/`: `stream.rs`, `recovery.rs`, `measurement.rs` and
//! `replica.rs`. This file holds what they share: the leader a case runs
//! against, a standby's stream, a worker's writes, and the twin a snapshot
//! and its events are held against. The standby's address stays beside the
//! leader's, which one comment names together.
#![expect(
    clippy::expect_used,
    reason = "the case's helpers run inside the test; a failure there is the test's failure"
)]

#[expect(
    dead_code,
    reason = "this binary dials loopback; the other helpers are the listeners' tests'"
)]
#[path = "../common/net.rs"]
mod net;
#[path = "../common/pair.rs"]
mod pair;

use std::{
    collections::BTreeMap,
    future::Future,
    io::ErrorKind,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use net::loopback;
use prost::Message as _;
use smg_grand_central::{
    engine::WorkerId,
    service::{
        controller::{Config, Controller, Error, LeasePlace, Listener, Scope, State},
        elector::{ElectionSettings, Elector, FileLease, TickOutcome},
        proto::{
            self, grand_central_client::GrandCentralClient, replica_frame::Frame,
            replication_event::Event as Sent, worker_event::Event,
        },
    },
    store::{HealthPolicy, MemoryStore, Registrant, WorkerFacts, WorkerStatus, WorkersStore},
    worker_host::{HealthAdapter, Members, Objects, SoleLead},
};
use smg_worker::health::port::{HealthSink, ProbeAnswer};
use tokio::sync::{mpsc, oneshot};
use tonic::{
    codegen::tokio_stream::wrappers::ReceiverStream, metadata::AsciiMetadataValue,
    transport::Channel, Code, Request, Status, Streaming,
};

#[path = "replication_cases/measurement.rs"]
mod measurement;
#[path = "replication_cases/recovery.rs"]
mod recovery;
#[path = "replication_cases/replica.rs"]
mod replica;
#[path = "replication_cases/stream.rs"]
mod stream;

/// Long against a loaded host: the leader of a case's start leads to its
/// end, unless the case makes it leave. A replica that left its lease
/// holds its takes back for the duration, so it takes none again inside
/// the case.
fn settings() -> ElectionSettings {
    let mut settings = ElectionSettings::default();
    settings.duration = Duration::from_secs(60);
    settings.renew_period = Duration::from_millis(500);
    settings.renew_deadline = Duration::from_secs(40);
    settings
}

fn listener(name: &str, scopes: &[Scope]) -> Listener {
    Listener::of(name.to_owned(), loopback(), scopes.to_vec(), None)
}

/// What a case reaches of the controller it runs against.
struct Leader {
    replication: SocketAddr,
    control: SocketAddr,
    gateway: SocketAddr,
    store: Arc<MemoryStore>,
    /// The holder of the controller's worker objects: what a worker the
    /// case registers is written through.
    workers: Arc<Objects<Members>>,
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
    let config = |port| {
        let mut configuration = Config::of(
            "replication-0".to_owned(),
            vec![
                listener("gateway", &[Scope::Gateway]),
                listener("control", &[Scope::Control]),
                {
                    let mut listening = listener("replication", &[Scope::Replication]);
                    listening.addr = SocketAddr::new(LEADER, port);
                    listening
                },
            ],
            loopback(),
        );
        configuration.shutdown_settle = Duration::from_millis(50);
        configuration.shutdown_grace = Duration::from_secs(5);
        configuration.decide_shards = 2;
        configuration.election.lease = place.clone();
        configuration.election.settings = settings();
        configuration.election.advertise = None;
        configuration
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
        workers: controller.workers(),
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

async fn client(addr: SocketAddr) -> GrandCentralClient<Channel> {
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("a uri")
        .connect()
        .await
        .expect("the listener accepts");
    GrandCentralClient::new(channel)
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
/// Its two thresholds are one, so each outcome a case hands the health
/// adapter moves its status ([`probed`]).
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
            failure_threshold: 1,
            success_threshold: 1,
            ..HealthPolicy::default()
        },
        ..WorkerFacts::default()
    }
}

/// What a case writes a worker through: a replica's holder of its worker
/// objects, and the store beside it. Every writer of a worker's facts
/// writes through the holder, so a first copy holds what the case wrote.
trait Hands {
    fn workers(&self) -> &Arc<Objects<Members>>;
    fn store(&self) -> &Arc<MemoryStore>;
}

impl Hands for Leader {
    fn workers(&self) -> &Arc<Objects<Members>> {
        &self.workers
    }

    fn store(&self) -> &Arc<MemoryStore> {
        &self.store
    }
}

impl Hands for (Arc<Objects<Members>>, Arc<MemoryStore>) {
    fn workers(&self) -> &Arc<Objects<Members>> {
        &self.0
    }

    fn store(&self) -> &Arc<MemoryStore> {
        &self.1
    }
}

/// A registration of `stated` through the holder, as a registration call
/// writes one: the worker's id.
fn lays(replica: &impl Hands, stated: WorkerFacts) -> WorkerId {
    let workers = replica.workers();
    let laid = workers.upsert(replica.store(), stated, Registrant::Control);
    laid.expect("the store took the registration").id()
}

/// One probe's outcome for the worker `id`, handed to the health adapter
/// of a host that leads, as the serve's prober hands one: the worker's
/// object moves and the store is told. Stale where the worker left.
fn probed(replica: &impl Hands, id: WorkerId, passed: bool) -> ProbeAnswer {
    let workers = replica.workers();
    let Some(object) = workers.object(id) else {
        return ProbeAnswer::Stale;
    };
    let Some(stands) = workers.registry().get(&object) else {
        return ProbeAnswer::Stale;
    };
    let (store, lead) = (Arc::clone(replica.store()), Arc::new(SoleLead));
    let health = HealthAdapter::new(Arc::clone(workers), store, lead, SoleLead::TERM);
    health.probe_outcome(&object, passed, stands.revision(), stands.status())
}

/// One worker and `more` watched writes of it: its status, turned each
/// time.
fn writes(leader: &impl Hands, more: u32) -> WorkerId {
    let id = lays(leader, facts("http://a:8000", 1, 0));
    for count in 0..more {
        turn(leader, id, count);
    }
    id
}

/// One watched write of the worker: its status, ready at an even `count`
/// and not ready at an odd one, by a probe that passed or failed.
fn turn(leader: &impl Hands, id: WorkerId, count: u32) {
    let moved = probed(leader, id, count.is_multiple_of(2));
    assert!(
        matches!(moved, ProbeAnswer::Moved { .. }),
        "the probe moved the worker: {moved:?}"
    );
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
/// keys, until `stop`.
fn churn(leader: &impl Hands, stop: &AtomicBool) -> u32 {
    let (workers, store) = (leader.workers(), leader.store());
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
        match (workers.read_row_under(&name), draw(8)) {
            (None, _) | (Some(_), 0) => {
                lives[index] += 1;
                lays(leader, facts(&name, lives[index], 0));
            }
            (Some(row), 1) => {
                workers.remove(store, row.id, row.revision);
            }
            (Some(row), 2) => {
                let more = row.facts.max_running.unwrap_or(0) + 1;
                lays(leader, facts(&name, lives[index], more));
            }
            (Some(row), 3) => {
                // By the table's status: the object's own leads it. This
                // thread is the one writer of the leader's rows, so the
                // table holds the row the holder read a line above.
                let told = store.get(row.id).expect("the table's row").status;
                probed(leader, row.id, told != WorkerStatus::Ready);
            }
            // The other half of the draws writes nothing.
            (Some(_), _) => {}
        }
        thread::sleep(Duration::from_micros(100));
    }
    rounds
}

/// Waits, within 20 s, until `replica`'s store wrote past position `seq`.
async fn written_past(replica: &impl Hands, seq: u64) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while replica.workers().checkpoint(replica.store()).at.seq <= seq {
        assert!(Instant::now() < deadline, "no write past {seq} in 20 s");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

fn count(n: usize) -> u64 {
    u64::try_from(n).expect("a count")
}

/// The leader's address and the standby's: one host, two addresses, so
/// both listen on one replication port as replicas configured alike do.
/// Linux routes the second with no alias. Another host refuses its bind:
/// a case that binds it is ignored there, with that reason.
const LEADER: IpAddr = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1));
const FOLLOWER: IpAddr = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2));
