//! A replica that does not lead follows (design 5.4): the standby's half of
//! the replication scope. It is restored from its leader's complete
//! snapshot, applies the batches of `Replicate`, and starts again from a
//! snapshot where the leader's log no longer continues its own.
//!
//! One invariant: a standby's tables and log are its leader's at the place
//! its log stands at. Each event lands by one write of the store under the
//! log's guard of the event's position ([`Log::adopt`]), so the twin's log
//! holds the leader's positions and terms, and an event that does not land
//! stops the stream ([`Halt`]). The applier is the one writer of the
//! replicated tables while a replica follows: its own load driver writes
//! nothing then.

use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use tokio::{
    sync::{mpsc, watch},
    time::{sleep_until, timeout},
};
use tonic::{
    codegen::tokio_stream::wrappers::ReceiverStream, transport::Channel, Code, Request, Status,
};
use tracing::{info, warn};

use super::{
    election::Standing,
    principal::ENTRY,
    replication::BEAT,
    service::{Leader, Shared},
    worker_wire::{self, Refused},
};
use crate::{
    proto::{
        self, policy_controller_client::PolicyControllerClient, replica_frame::Frame,
        replication_event::Event as Sent,
    },
    replication::{Log, Lost},
    store::{Kinds, LeasesStore, LoadSink, MemoryStore, Position},
    view::fingerprint,
};

/// Why a standby stopped applying its leader's stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Halt {
    /// The standby does not stand where the stream continues: its log
    /// refused the place, or the event did not fit its tables. A snapshot
    /// puts it there.
    Lost(Position),
    /// A value this build cannot read, by its path: the leader runs a
    /// build this one does not follow, and no snapshot changes that.
    Unread(Refused),
}

/// The standby's applier over one store and the log it journals to.
pub(crate) struct Follower {
    store: Arc<MemoryStore>,
    log: Arc<Log>,
}

impl Follower {
    pub(crate) fn new(store: Arc<MemoryStore>, log: Arc<Log>) -> Self {
        Self { store, log }
    }

    /// Where the standby stands: the place its next hello names. A standby
    /// that applied nothing names no place of any run, zero for both: its
    /// log stands at the place its own run drew, and its leader continues
    /// it from the leader's first write.
    pub(crate) fn place(&self) -> Position {
        let head = self.log.head(Kinds::All);
        if self.log.first(Kinds::All) == Some(head) {
            Position::default()
        } else {
            head
        }
    }

    /// The store restored from the leader's complete snapshot, read at
    /// `now`: the place the standby stands at after it. [`Halt::Lost`]
    /// where the log refuses the snapshot's place (a history that is not
    /// ahead of its own), the store as it was.
    pub(crate) fn restore(
        &self,
        snapshot: &proto::Snapshot,
        now: Instant,
    ) -> Result<Position, Halt> {
        let cut = worker_wire::told_cut(snapshot, now).map_err(Halt::Unread)?;
        let at = cut.at;
        if self.store.restore_from(cut) {
            Ok(at)
        } else {
            Err(Halt::Lost(at))
        }
    }

    /// One batch applied at `now`: each event at its position under the
    /// batch's term, by the store's one write for its kind. A load row
    /// stands with the instant of its apply. The first event that does not
    /// land stops the batch; the events before it stand, each at its place.
    /// A standby that applied nothing takes its leader's first position, a
    /// term's start, wherever the leader's run drew it ([`Log::join`]); no
    /// other event lands on it.
    pub(crate) fn apply(&self, batch: &proto::ReplicationBatch, now: Instant) -> Result<(), Halt> {
        for (event, seq) in batch.events.iter().zip(batch.first_seq..) {
            let at = Position {
                seq,
                term: batch.term,
            };
            let adopting = match self.log.adopt(at) {
                Ok(adopting) => adopting,
                // A standby that applied nothing: its log stands at the
                // place its own run drew, and its leader's history starts
                // with a term's start, which writes no row. The log joins
                // there first: told the term with no place to journal it
                // at, it would start a lead of the store's own.
                Err(Lost) if matches!(event.event, Some(Sent::TermStart(_))) => {
                    self.log.join(at).map_err(|Lost| Halt::Lost(at))?;
                    self.store.set_term(at.term);
                    continue;
                }
                Err(Lost) => return Err(Halt::Lost(at)),
            };
            match &event.event {
                Some(Sent::Membership(told)) => {
                    let (epoch, change, row) =
                        worker_wire::told_change(told, now).map_err(Halt::Unread)?;
                    self.store.follow(epoch, change, row);
                }
                Some(Sent::Load(told)) => {
                    let (id, rank, row, source) =
                        worker_wire::told_load(told).map_err(Halt::Unread)?;
                    self.store.record(id, rank, row, now, source);
                }
                Some(Sent::TermStart(_)) => self.store.set_term(at.term),
                // A kind this build does not apply, and one it does not know.
                Some(_) | None => {
                    return Err(Halt::Unread(Refused {
                        path: "batch.events",
                        why: "an event this build does not apply",
                    }));
                }
            }
            if !adopting.landed() {
                return Err(Halt::Lost(at));
            }
        }
        Ok(())
    }
}

/// How far behind its leader a standby is, as its last applied frame says
/// it: the loop writes it and `Leader` reads it.
#[derive(Debug)]
pub(crate) struct Lag {
    origin: Instant,
    /// The last applied frame, in microseconds past the origin plus one;
    /// `0` before any.
    frame: AtomicU64,
}

impl Lag {
    pub(crate) fn new() -> Self {
        Self {
            origin: Instant::now(),
            frame: AtomicU64::new(0),
        }
    }

    /// A frame of the leader's stream, or its snapshot, was applied at
    /// `at`.
    pub(crate) fn applied(&self, at: Instant) {
        let micros = at.saturating_duration_since(self.origin).as_micros();
        let micros = u64::try_from(micros).unwrap_or(u64::MAX - 1);
        self.frame.store(micros + 1, Ordering::Relaxed);
    }

    /// The age at `now` of the last applied frame; `None` before any.
    pub(crate) fn age(&self, now: Instant) -> Option<Duration> {
        let frame = self.frame.load(Ordering::Relaxed).checked_sub(1)?;
        let applied = self.origin + Duration::from_micros(frame);
        Some(now.saturating_duration_since(applied))
    }
}

/// The lag a standby reads synced under: one load poll interval, the
/// period its load rows are no staler than then, and never under two of
/// the leader's heartbeats, which an idle stream's lag reaches by itself.
/// The lag is the first of the counts synced reads (design 5.4): the live
/// ranks and the generation narrow the same bit with their steps.
pub(crate) fn synced_within(load_poll_interval: Duration) -> Duration {
    load_poll_interval.max(2 * BEAT)
}

/// How long a dial may take; a leader that does not accept within it is
/// asked again.
const DIAL: Duration = Duration::from_secs(2);

/// The wait after the first stream that applied nothing. It doubles a
/// stream, up to [`MOST`].
const FIRST: Duration = Duration::from_millis(100);

/// The longest wait between two streams.
const MOST: Duration = Duration::from_secs(5);

/// How many of the leader's heartbeats a stream may bring no frame for
/// before it is left. A leader that froze, or a host that is gone with no
/// end of its connection, sends nothing and ends nothing.
const QUIET_BEATS: u32 = 3;

/// How many times a stream's quiet bound the leader's snapshot may take to
/// arrive whole.
const SNAPSHOT_QUIETS: u32 = 10;

/// How often a standby asks its leader for a snapshot at most. A cut holds
/// the leader's writers for its length and nothing else paces the call: a
/// standby that keeps losing its place asks one a beat.
const RESYNC: Duration = BEAT;

/// The pace of a standby's snapshots: when the next may be asked.
struct Pace {
    every: Duration,
    /// When the last snapshot was asked, or is to be.
    due: Mutex<Option<tokio::time::Instant>>,
}

impl Pace {
    fn new(every: Duration) -> Self {
        Self {
            every,
            due: Mutex::new(None),
        }
    }

    /// Resolves when a snapshot may be asked: at once where the last was
    /// asked `every` ago or longer, else when that long has passed since.
    async fn wait(&self) {
        let due = {
            let mut due = self.due.lock();
            let now = tokio::time::Instant::now();
            let next = due.map_or(now, |last| (last + self.every).max(now));
            *due = Some(next);
            next
        };
        sleep_until(due).await;
    }
}

/// The wait before stream `tries` + 1 of a streak that applied nothing:
/// between half the doubled wait and all of it, the half a replica waits
/// being its identity's own, so the standbys of one leader do not ask it
/// together.
fn backoff(tries: u32, identity: &str) -> Duration {
    let whole = FIRST.saturating_mul(1 << tries.min(8)).min(MOST);
    let half = whole / 2;
    let millis = u64::try_from(half.as_millis()).unwrap_or(u64::MAX);
    let own = fingerprint(format!("{identity}/{tries}").as_bytes()) % (millis + 1);
    half + Duration::from_millis(own)
}

/// A streak of streams that applied nothing: how long it is, and the reason
/// said last.
#[derive(Debug, Default)]
struct Streak {
    tries: u32,
    said: Option<String>,
}

impl Streak {
    /// A stream ended, with `why` if it said one. A stream that applied
    /// ends the streak first, so its reason is said anew if it returns.
    /// The reason to say: `why`, unless it is the one said last.
    fn ended(&mut self, applied: bool, why: Option<String>) -> Option<&str> {
        if applied {
            *self = Self::default();
        }
        let why = why.filter(|why| self.said.as_ref() != Some(why))?;
        self.said = Some(why);
        self.said.as_deref()
    }

    /// The wait before the streak's next stream, longer a stream
    /// ([`backoff`]).
    fn wait(&mut self, identity: &str) -> Duration {
        let wait = backoff(self.tries, identity);
        self.tries = self.tries.saturating_add(1);
        wait
    }
}

/// Waits `wait` at most between two streams. A wait for the lease to be
/// read again (`lease`) ends at the election's next step; a streak's wait
/// ends at the step that names another holder.
async fn pause(steps: &mut watch::Receiver<Standing>, lease: bool, wait: Duration) {
    let holder = steps.borrow_and_update().holder.clone();
    let moved = async {
        while steps.changed().await.is_ok() {
            if lease || steps.borrow().holder != holder {
                return;
            }
        }
        std::future::pending::<()>().await;
    };
    tokio::select! {
        () = moved => {}
        () = tokio::time::sleep(wait) => {}
    }
}

/// What the loop does after a stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Next {
    /// The standby was restored: it joins again at once, at its new place.
    Join,
    /// The lease no longer names the stream's leader: the loop reads it
    /// again at once.
    Turn,
    /// The leader asked for patience, or could not be reached: the stream
    /// is tried again after the streak's wait.
    Wait,
    /// The replica asked is not the leader, or this one leads: the lease
    /// is read again at the election's next step.
    Lease,
}

/// How a stream ended: what the loop does next, whether a frame of it was
/// applied, and why it ended.
struct Ended {
    next: Next,
    applied: bool,
    /// Why it ended, where that is a failure to say.
    why: Option<String>,
}

/// A replica's standby loop: while the replica does not lead, it follows
/// the holder its lease names.
pub(crate) struct Standby {
    shared: Arc<Shared>,
    follower: Follower,
    /// The port of this replica's own replication listener. The leader is
    /// dialed at its advertised host under this port: replicas are
    /// configured alike.
    port: u16,
    /// How long a stream may bring no frame: [`QUIET_BEATS`] of the
    /// leader's heartbeats.
    quiet: Duration,
    /// When this replica may ask its next snapshot ([`RESYNC`]).
    pace: Pace,
}

impl Standby {
    pub(crate) fn new(shared: Arc<Shared>, port: u16) -> Self {
        let follower = Follower::new(Arc::clone(&shared.store), Arc::clone(&shared.feed.log));
        Self {
            shared,
            follower,
            port,
            quiet: QUIET_BEATS * BEAT,
            pace: Pace::new(RESYNC),
        }
    }

    /// Follows until `stop` reads `true`: one stream after another, each
    /// against the holder the lease names at that moment, and none while
    /// this replica leads or nobody is named. A streak of streams that
    /// applied nothing is said once, by its reason, and waits longer a
    /// stream ([`backoff`]); the lease naming another holder ends the wait,
    /// and the stream wherever it stands, the leader's answer not waited
    /// for.
    pub(crate) async fn run(self, mut stop: watch::Receiver<bool>) {
        let mut steps = self.shared.standing.clone();
        let mut joined = false;
        let mut streak = Streak::default();
        loop {
            let leading = self.shared.fence.leader_term(Instant::now());
            let leader = self.shared.leader(leading);
            let next = if leader.is_leader || leader.address.is_empty() {
                Next::Lease
            } else {
                let applied = AtomicBool::new(false);
                let ended = tokio::select! {
                    ended = self.stream(&leader, joined, &applied) => ended,
                    next = self.turned(&mut steps, &leader) => Ended {
                        next,
                        applied: applied.load(Ordering::Relaxed),
                        why: None,
                    },
                    _ = stop.wait_for(|stop| *stop) => return,
                };
                if let Some(why) = streak.ended(ended.applied, ended.why) {
                    warn!(
                        controller_id = %self.shared.controller_id,
                        leader = %leader.controller_id,
                        address = %self.target(&leader.address),
                        why = %why,
                        "this standby does not follow its leader; it reads not synced until it does"
                    );
                }
                ended.next
            };
            joined = next == Next::Join;
            let wait = match next {
                Next::Join | Next::Turn => continue,
                Next::Lease => MOST,
                Next::Wait => streak.wait(&self.shared.identity),
            };
            tokio::select! {
                () = pause(&mut steps, next == Next::Lease, wait) => {}
                _ = stop.wait_for(|stop| *stop) => return,
            }
        }
    }

    /// Resolves when the stream against `leader` is to be left for the
    /// lease's sake: with [`Next::Turn`] once the lease names another
    /// holder, another term of the same one or this replica, with
    /// [`Next::Lease`] once the election is gone.
    async fn turned(&self, steps: &mut watch::Receiver<Standing>, leader: &Leader) -> Next {
        loop {
            if steps.changed().await.is_err() {
                return Next::Lease;
            }
            let leading = self.shared.fence.leader_term(Instant::now());
            let named = self.shared.leader(leading);
            if named.controller_id != leader.controller_id
                || named.address != leader.address
                || named.term != leader.term
            {
                return Next::Turn;
            }
        }
    }

    /// A streak's reason for a leader that says nothing within the bound.
    fn silent(&self, what: &str) -> String {
        format!("the leader {what} for {:?}", self.quiet)
    }

    /// Where the leader advertised at `address` is dialed: its host, at
    /// this replica's own replication port.
    fn target(&self, address: &str) -> String {
        let host = address.rsplit_once(':').map_or(address, |(host, _)| host);
        format!("http://{host}:{}", self.port)
    }

    /// `message` as a call that states this replica's name.
    fn stated<T>(&self, message: T) -> Request<T> {
        let mut request = Request::new(message);
        if let Ok(name) = self.shared.controller_id.parse() {
            request.metadata_mut().insert(ENTRY, name);
        }
        request
    }
}

/// How many frames of its own a stream holds unsent: the hello, then the
/// acknowledgements, each of which covers the ones before it.
const FRAMES: usize = 8;

fn frame(frame: Frame) -> proto::ReplicaFrame {
    proto::ReplicaFrame { frame: Some(frame) }
}

impl Ended {
    /// A stream the leader refused or broke. The code says the wait: a
    /// replica that is not the leader, one deposed and one fenced mid-stream
    /// send the standby back to its lease; any other is asked again.
    fn refused(status: &Status, applied: bool) -> Self {
        let next = match status.code() {
            Code::FailedPrecondition | Code::Aborted => Next::Lease,
            _ => Next::Wait,
        };
        Self {
            next,
            applied,
            why: Some(format!("{:?}: {}", status.code(), status.message())),
        }
    }

    /// A stream that ended for `why` and is tried again after a wait.
    fn waits(applied: bool, why: String) -> Self {
        Self {
            next: Next::Wait,
            applied,
            why: Some(why),
        }
    }

    /// A stream this replica left because it leads: nothing to say.
    const fn leads(applied: bool) -> Self {
        Self {
            next: Next::Lease,
            applied,
            why: None,
        }
    }
}

impl Standby {
    /// One stream's life against `leader`: the dial, the hello at the log's
    /// head, then every batch applied and acknowledged until the stream
    /// ends. Where the leader's log does not continue this one (a
    /// `resync_required` batch, an event that does not land) the standby is
    /// restored from the leader's complete snapshot and joins again;
    /// `joined` says this stream is such a join, and `applied` is set once
    /// a frame of it landed: the loop reads it where it leaves the stream
    /// itself. A leader that opens no stream, or sends no frame, within
    /// the quiet bound is left and asked again.
    async fn stream(&self, leader: &Leader, joined: bool, applied: &AtomicBool) -> Ended {
        let dialed = match Channel::from_shared(self.target(&leader.address)) {
            Ok(endpoint) => endpoint
                .connect_timeout(DIAL)
                .tcp_nodelay(true)
                .connect()
                .await
                .map_err(|error| error.to_string()),
            Err(error) => Err(error.to_string()),
        };
        let mut client = match dialed {
            Ok(channel) => PolicyControllerClient::new(channel),
            Err(error) => return Ended::waits(false, format!("the dial failed: {error}")),
        };
        let place = self.follower.place();
        let hello = proto::ReplicaHello {
            replica_id: self.shared.controller_id.clone(),
            last_applied_seq: place.seq,
            term_seen: leader.term,
            max_backlog: 0,
            last_applied_term: place.term,
        };
        let (frames, sent) = mpsc::channel(FRAMES);
        // The hello is in the channel before the call opens.
        let _ = frames.try_send(frame(Frame::Hello(hello)));
        let opened = client.replicate(self.stated(ReceiverStream::new(sent)));
        let mut batches = match timeout(self.quiet, opened).await {
            Ok(Ok(opened)) => opened.into_inner(),
            Ok(Err(status)) => return Ended::refused(&status, false),
            Err(_quiet) => return Ended::waits(false, self.silent("opened no stream")),
        };
        loop {
            let landed = applied.load(Ordering::Relaxed);
            let batch = match timeout(self.quiet, batches.message()).await {
                Ok(Ok(Some(batch))) => batch,
                Ok(Ok(None)) => {
                    return Ended::waits(landed, "the leader ended the stream".to_owned())
                }
                Ok(Err(status)) => return Ended::refused(&status, landed),
                Err(_quiet) => return Ended::waits(landed, self.silent("sent no frame")),
            };
            let now = Instant::now();
            let lost = batch.resync_required
                || match self.follower.apply(&batch, now) {
                    Ok(()) => false,
                    Err(Halt::Lost(_)) => true,
                    Err(Halt::Unread(refused)) => return Ended::waits(landed, unread(refused)),
                };
            if lost {
                return self.restored(&mut client, landed, joined).await;
            }
            if !landed {
                let place = self.follower.place();
                info!(
                    controller_id = %self.shared.controller_id,
                    leader = %leader.controller_id,
                    seq = place.seq,
                    term = place.term,
                    "following the leader's log"
                );
            }
            applied.store(true, Ordering::Relaxed);
            self.shared.lag.applied(now);
            if !batch.events.is_empty() {
                // An acknowledgement names the place reached: one the
                // channel has no room for is covered by the next.
                let seq = self.follower.place().seq;
                let _ = frames.try_send(frame(Frame::Ack(proto::ReplicationAck { seq })));
            }
        }
    }

    /// The standby restored from its leader's complete snapshot, after
    /// which it joins again at the snapshot's place. A join that needs a
    /// second snapshot before it applied a frame waits like a stream that
    /// applied nothing. A snapshot is asked a beat after the last at the
    /// soonest ([`RESYNC`]), the stream left standing meanwhile. No snapshot
    /// is taken by a replica that leads by now, and the log refuses one that
    /// is not ahead of its own history.
    async fn restored(
        &self,
        client: &mut PolicyControllerClient<Channel>,
        applied: bool,
        joined: bool,
    ) -> Ended {
        self.pace.wait().await;
        if self.shared.fence.leader_term(Instant::now()).is_some() {
            return Ended::leads(applied);
        }
        let request = self.stated(proto::SnapshotRequest::default());
        let bound = self.quiet.saturating_mul(SNAPSHOT_QUIETS);
        let snapshot = match timeout(bound, client.get_snapshot(request)).await {
            Ok(Ok(snapshot)) => snapshot.into_inner(),
            Ok(Err(status)) => return Ended::refused(&status, applied),
            Err(_quiet) => {
                let why = format!("the leader's snapshot did not arrive in {bound:?}");
                return Ended::waits(applied, why);
            }
        };
        let now = Instant::now();
        match self.follower.restore(&snapshot, now) {
            Ok(place) => {
                self.shared.lag.applied(now);
                info!(
                    controller_id = %self.shared.controller_id,
                    seq = place.seq,
                    term = place.term,
                    rows = snapshot.workers.len(),
                    "restored from the leader's snapshot"
                );
                if joined && !applied {
                    let why = "restored twice with no frame between".to_owned();
                    return Ended::waits(false, why);
                }
                Ended {
                    next: Next::Join,
                    applied: true,
                    why: None,
                }
            }
            Err(Halt::Lost(at)) => Ended {
                next: Next::Lease,
                applied,
                why: Some(format!(
                    "the leader's snapshot stands at {} under term {}, which this replica's log does not follow",
                    at.seq, at.term
                )),
            },
            Err(Halt::Unread(refused)) => Ended::waits(applied, unread(refused)),
        }
    }
}

/// A value of the leader's stream this build cannot read, as a streak's
/// reason.
fn unread(refused: Refused) -> String {
    format!(
        "the leader's {} is {}: this build does not follow that leader",
        refused.path, refused.why
    )
}

#[cfg(test)]
mod tests {
    use std::{future::pending, io, net::SocketAddr};

    use parking_lot::Mutex;
    use tokio::{net::TcpListener, sync::oneshot, task::JoinSet};
    use tonic::{
        transport::{server::TcpIncoming, Error as Transport, Server},
        Response, Streaming,
    };

    use super::*;
    use crate::{
        controller::{
            election::LeasePlace,
            replication::{journaled, Replication},
        },
        elector::{ElectionSettings, Elector, Tick},
        proto::policy_controller_server::{PolicyController, PolicyControllerServer},
        replication::{Entry, Event},
        store::{LoadSource, WorkerFacts, WorkerStatus, WorkersStore},
        view::{LoadRow, WireModes},
    };

    type Replica = (Arc<MemoryStore>, Arc<Log>);

    /// A replica's store and the log it journals to under a shared lease:
    /// the log's places start where this replica's run drew.
    fn replica() -> Replica {
        let (store, feed) = journaled(&LeasePlace::File("shared".into()));
        (store, feed.log)
    }

    /// The place before a log's first write, where a standby that applied
    /// nothing is continued from.
    fn start(log: &Log) -> Position {
        log.first(Kinds::All).expect("every write is retained")
    }

    /// A leader's store and log under term 1, and a standby's applier over
    /// its own.
    fn pair() -> (Replica, Follower, Replica) {
        let leader = replica();
        leader.0.set_term(1);
        let standby = replica();
        let follower = Follower::new(Arc::clone(&standby.0), Arc::clone(&standby.1));
        (leader, follower, standby)
    }

    fn facts(key: &str, incarnation: u64) -> WorkerFacts {
        WorkerFacts {
            key: key.to_owned(),
            engine: key.to_owned(),
            incarnation,
            models: vec!["m".to_owned()],
            dp_size: 1,
            wire_modes: WireModes::GRPC,
            ..WorkerFacts::default()
        }
    }

    fn running(count: u32) -> LoadRow {
        LoadRow {
            running: count,
            ..LoadRow::default()
        }
    }

    /// Every entry of `log` after `place`.
    fn entries(log: &Log, mut place: Position) -> Vec<Entry> {
        let (mut all, mut out) = (Vec::new(), Vec::new());
        loop {
            log.read(Kinds::All, place, 64, &mut out)
                .expect("the place is held");
            let Some(last) = out.last() else {
                return all;
            };
            place = last.at;
            all.append(&mut out);
        }
    }

    /// One entry as the batch a leader sends it in at `now`.
    fn batch(entry: &Entry, now: Instant) -> proto::ReplicationBatch {
        proto::ReplicationBatch {
            first_seq: entry.at.seq,
            term: entry.at.term,
            events: vec![worker_wire::replication(entry, now)],
            resync_required: false,
        }
    }

    /// The standby applies what the leader's log holds after its place, a
    /// standby that applied nothing from the leader's first write.
    fn follow(follower: &Follower, leader: &Log, now: Instant) {
        let place = match follower.place() {
            nothing if nothing == Position::default() => start(leader),
            place => place,
        };
        for entry in entries(leader, place) {
            assert_eq!(
                follower.apply(&batch(&entry, now), now),
                Ok(()),
                "{entry:?}"
            );
        }
    }

    /// What a replica holds, without the instants: its place, its epoch,
    /// its rows and its load rows.
    fn held(store: &MemoryStore) -> impl PartialEq + std::fmt::Debug {
        let cut = store.cut(Kinds::All);
        let rows: Vec<_> = cut
            .rows
            .into_iter()
            .map(|row| (row.id, row.facts, row.status, row.revision))
            .collect();
        let loads: Vec<_> = cut
            .loads
            .into_iter()
            .map(|(id, rank, sample)| (id, rank, sample.row))
            .collect();
        (cut.at, cut.epoch, rows, loads, store.term())
    }

    /// A log's entries by place, each as its kind and what it names.
    fn told(log: &Log) -> Vec<(Position, String)> {
        entries(log, start(log))
            .into_iter()
            .map(|entry| {
                let what = match entry.event {
                    Event::Member { epoch, change, .. } => format!("{epoch} {change:?}"),
                    Event::Load { id, rank, .. } => format!("load {id:?} {rank}"),
                    Event::Term => "term".to_owned(),
                };
                (entry.at, what)
            })
            .collect()
    }

    /// Writes under the first term: registrations, a status change, a load
    /// row of each source, a relabel and a resize.
    fn first(leader: &MemoryStore, at: Instant) {
        let revision = |id| WorkersStore::get(leader, id).expect("the row").revision;
        let a = leader.upsert(facts("http://a:8000", 1)).id();
        let b = leader.upsert(facts("http://b:8000", 1)).id();
        leader.set_status(a, WorkerStatus::Ready, revision(a));
        leader.record(a, 0, running(3), at, LoadSource::Stream);
        leader.report(&[(b, 0, Some(running(5)))], at, LoadSource::Poll);
        let mut labelled = facts("http://a:8000", 1);
        labelled.labels.insert("tier".to_owned(), "gold".to_owned());
        leader.upsert(labelled);
        let wide = WorkerFacts {
            dp_size: 2,
            ..facts("http://b:8000", 1)
        };
        leader.upsert(wide);
    }

    /// What follows: a new life of a ready worker (its status comes over),
    /// a removal, the next term's start and a registration under it.
    fn rest(leader: &MemoryStore, at: Instant) {
        let again = leader.upsert(facts("http://a:8000", 2)).id();
        leader.record(again, 0, running(7), at, LoadSource::Stream);
        let b = leader.by_key("http://b:8000").expect("the row");
        leader.remove(b.id, b.revision);
        leader.set_term(2);
        leader.upsert(facts("http://c:8000", 1));
    }

    #[test]
    fn a_standby_fed_the_leaders_log_holds_its_tables_and_its_log() {
        let ((leader, led), follower, (standby, followed)) = pair();
        // It applied nothing: it names no place of any run, and its log
        // stands where its own run drew.
        assert_eq!(follower.place(), Position::default());
        assert_eq!(start(&followed), followed.head(Kinds::All));
        let at = Instant::now();
        first(&leader, at);
        follow(&follower, &led, at);
        // The join told the store its leader's term: under the first term
        // the standby holds what its leader holds, the term with it.
        assert_eq!(
            format!("{:?}", held(&standby)),
            format!("{:?}", held(&leader))
        );
        rest(&leader, at);
        follow(&follower, &led, at);
        // Its log starts where its leader's does: it serves a standby that
        // applied nothing from there when it leads.
        assert_eq!(start(&followed), start(&led));
        assert_eq!(
            format!("{:?}", held(&standby)),
            format!("{:?}", held(&leader))
        );
        assert_eq!(told(&followed), told(&led));
        assert_eq!(follower.place(), led.head(Kinds::All));
        assert!(told(&led).len() > 10, "{:?}", told(&led));
    }

    /// The leader's complete snapshot at `at`, as its replication scope
    /// serves it under the term its store wrote.
    fn snapshot((store, log): &Replica, at: Instant) -> proto::Snapshot {
        let mut elector =
            Elector::alone("controller-0@10.0.0.7:30100", ElectionSettings::default())
                .expect("the default settings are in order");
        let Ok(Tick::Acquired { term }) = elector.tick(at) else {
            panic!("a free lease is taken at the first tick");
        };
        Replication::new(Arc::clone(store), Arc::clone(log), elector.fence())
            .snapshot(term, "controller-0", at)
            .expect("a snapshot")
    }

    #[test]
    fn a_standby_restored_from_the_snapshot_continues_from_its_place() {
        let (leader, follower, (standby, _followed)) = pair();
        let at = Instant::now();
        first(&leader.0, at);
        // The standby held a row of its own, under no term.
        standby.upsert(facts("http://stale:8000", 1));
        let first = snapshot(&leader, at);
        let place = follower.restore(&first, at);
        assert_eq!(place, Ok(leader.1.head(Kinds::All)));
        assert_eq!(held(&standby), held(&leader.0));
        rest(&leader.0, at);
        follow(&follower, &leader.1, at);
        assert_eq!(held(&standby), held(&leader.0));
        // A snapshot of the term it followed before is a history it left:
        // refused, the store as it was.
        let left = Position {
            seq: first.replication_seq,
            term: first.replication_term,
        };
        assert_eq!(follower.restore(&first, at), Err(Halt::Lost(left)));
        assert_eq!(held(&standby), held(&leader.0));
    }

    #[test]
    fn an_event_that_does_not_land_halts_the_batch_where_it_stands() {
        let ((leader, led), follower, (standby, _followed)) = pair();
        let at = Instant::now();
        let a = leader.upsert(facts("http://a:8000", 1)).id();
        leader.record(a, 0, running(1), at, LoadSource::Stream);
        let sent = entries(&led, start(&led));
        let [term, registered, load] = &sent[..] else {
            panic!("the term's start, a registration and a load row, not {sent:?}");
        };
        // A history starts with its term's start: the registration before
        // it is no place a standby that applied nothing joins at.
        let lost = Err(Halt::Lost(registered.at));
        assert_eq!(follower.apply(&batch(registered, at), at), lost);
        assert_eq!(follower.place(), Position::default());
        assert_eq!(follower.apply(&batch(term, at), at), Ok(()));
        // The term's start again: a replay, and no second join.
        let replay = Err(Halt::Lost(term.at));
        assert_eq!(follower.apply(&batch(term, at), at), replay);
        // A load row of a worker the standby does not hold.
        let stray = proto::ReplicationBatch {
            first_seq: registered.at.seq,
            ..batch(load, at)
        };
        assert_eq!(follower.apply(&stray, at), lost);
        // An event of no kind, and one of a kind this build does not apply.
        for event in [None, Some(Sent::Health(proto::HealthRow::default()))] {
            let unread = proto::ReplicationBatch {
                events: vec![proto::ReplicationEvent { event }],
                ..batch(registered, at)
            };
            assert!(matches!(
                follower.apply(&unread, at),
                Err(Halt::Unread(refused)) if refused.path == "batch.events"
            ));
        }
        // A row this build cannot read is refused by its path.
        let mut unknown = batch(registered, at);
        if let Some(Sent::Membership(member)) = &mut unknown.events[0].event {
            if let Some(proto::worker_event::Event::Registered(row)) = &mut member.event {
                row.status = 99;
            }
        }
        assert!(matches!(
            follower.apply(&unknown, at),
            Err(Halt::Unread(refused)) if refused.path == "row.status"
        ));
        // None of them was applied: the standby stands at the term's start.
        assert_eq!((follower.place(), standby.epoch()), (term.at, 0));
        // The first event of a batch lands and the second does not fit:
        // the first stands at its place.
        let twice = proto::ReplicationBatch {
            events: vec![worker_wire::replication(registered, at); 2],
            ..batch(registered, at)
        };
        assert_eq!(follower.apply(&twice, at), Err(Halt::Lost(load.at)));
        assert_eq!((follower.place(), standby.epoch()), (registered.at, 1));
    }

    #[test]
    fn the_lag_is_the_age_of_the_last_applied_frame() {
        let lag = Lag::new();
        let at = Instant::now();
        let millis = |now| lag.age(now).map(|age| age.as_millis());
        assert_eq!(millis(at), None);
        lag.applied(at);
        assert_eq!(millis(at), Some(0));
        assert_eq!(millis(at + Duration::from_millis(250)), Some(250));
        lag.applied(at + Duration::from_millis(200));
        assert_eq!(millis(at + Duration::from_millis(250)), Some(50));
    }

    #[test]
    fn a_streak_says_a_reason_once_and_starts_again_after_a_stream_that_applied() {
        let mut streak = Streak::default();
        let waits = |streak: &mut Streak| [streak.wait("a"), streak.wait("a"), streak.wait("a")];
        let first = waits(&mut streak);
        assert_eq!(first, [backoff(0, "a"), backoff(1, "a"), backoff(2, "a")]);
        // A reason is said once while it stays, and another one is said.
        let refused = || Some("refused".to_owned());
        assert_eq!(streak.ended(false, refused()), Some("refused"));
        assert_eq!(streak.ended(false, refused()), None);
        assert_eq!(streak.ended(false, None), None);
        assert_eq!(
            streak.ended(false, Some("silent".to_owned())),
            Some("silent")
        );
        assert_eq!(streak.wait("a"), backoff(3, "a"));
        // A stream that applied ends the streak: its reason is said anew
        // and the waits start again.
        assert_eq!(
            streak.ended(true, Some("silent".to_owned())),
            Some("silent")
        );
        assert_eq!(waits(&mut streak), first);
    }

    #[tokio::test(start_paused = true)]
    async fn a_streaks_wait_ends_at_the_step_that_names_another_holder() {
        let stands = |holder: &str, term| Standing {
            holder: holder.to_owned(),
            term,
            looked: true,
        };
        let (standing, mut steps) = watch::channel(stands("a@h:1", 1));
        // A step that names the same holder: the wait runs out.
        let began = tokio::time::Instant::now();
        let same = async { standing.send_replace(stands("a@h:1", 1)) };
        tokio::join!(pause(&mut steps, false, MOST), same);
        assert_eq!(began.elapsed(), MOST);
        // A step that names another holder ends it, and any step ends a
        // wait for the lease.
        let began = tokio::time::Instant::now();
        let other = async { standing.send_replace(stands("b@h:1", 2)) };
        tokio::join!(pause(&mut steps, false, MOST), other);
        let again = async { standing.send_replace(stands("b@h:1", 2)) };
        tokio::join!(pause(&mut steps, true, MOST), again);
        assert_eq!(began.elapsed(), Duration::ZERO);
    }

    #[test]
    fn a_streaks_wait_doubles_to_its_bound_and_half_of_it_is_the_replicas_own() {
        let replicas = ["controller-0@10.0.0.7:30100", "controller-1@10.0.0.8:30100"];
        for tries in 0..40 {
            let whole = FIRST.saturating_mul(1 << tries.min(8)).min(MOST);
            for identity in replicas {
                let wait = backoff(tries, identity);
                assert!(whole / 2 <= wait && wait <= whole, "{tries}: {wait:?}");
                assert_eq!(wait, backoff(tries, identity));
            }
        }
        assert_eq!(
            backoff(0, replicas[0]).as_millis() / 50,
            1,
            "the first wait"
        );
        assert!(backoff(39, replicas[0]) >= MOST / 2);
        // Two standbys of one leader do not wait alike along a streak.
        assert!((0..8).any(|tries| backoff(tries, replicas[0]) != backoff(tries, replicas[1])));
    }

    #[test]
    fn a_refusals_code_says_which_wait_follows() {
        for (status, next) in [
            (
                Status::unavailable("not yet written under term 4"),
                Next::Wait,
            ),
            (Status::resource_exhausted("above one message"), Next::Wait),
            (Status::unauthenticated("no principal"), Next::Wait),
            (Status::failed_precondition("not the leader"), Next::Lease),
            (Status::aborted("the fence closed"), Next::Lease),
        ] {
            let ended = Ended::refused(&status, true);
            assert_eq!((ended.next, ended.applied), (next, true), "{status}");
            let why = ended.why.expect("a refusal is said");
            assert!(why.contains(status.message()), "{why}");
        }
    }

    #[test]
    fn a_standby_is_synced_under_one_load_poll_interval_and_never_under_two_beats() {
        assert_eq!(
            synced_within(Duration::from_secs(10)),
            Duration::from_secs(10)
        );
        assert_eq!(synced_within(Duration::from_secs(1)), 2 * BEAT);
        assert_eq!(synced_within(Duration::ZERO), 2 * BEAT);
    }

    #[tokio::test(start_paused = true)]
    async fn a_snapshot_is_asked_a_beat_after_the_last_at_the_soonest() {
        let pace = Pace::new(RESYNC);
        let began = tokio::time::Instant::now();
        pace.wait().await;
        assert_eq!(began.elapsed(), Duration::ZERO, "the first at once");
        // A second and a third in a row: a beat each.
        pace.wait().await;
        assert_eq!(began.elapsed(), BEAT);
        pace.wait().await;
        assert_eq!(began.elapsed(), 2 * RESYNC);
        // One asked later than a beat after the last waits for nothing.
        tokio::time::sleep(3 * RESYNC).await;
        let late = tokio::time::Instant::now();
        pace.wait().await;
        assert_eq!(late.elapsed(), Duration::ZERO);
        // The one right after it waits a beat from then: the late ask is
        // the last asked, not the instant it was due.
        pace.wait().await;
        assert_eq!(late.elapsed(), RESYNC);
    }

    /// A measurement, run by name with `--ignored --nocapture` in the
    /// release profile: what the applier costs an event (a load row, a
    /// status change) in batches of 256 as the leader sends them, and what a
    /// restore from the complete snapshot costs at 512 and 4,096 workers of
    /// eight load rows each. The wire's own decode is the transport's and
    /// is not in it.
    #[test]
    #[ignore = "a measurement: run by name"]
    fn the_standbys_numbers() {
        use io::Write as _;
        let mut out = io::stdout();
        for workers in [512u32, 4_096] {
            let (leader, follower, _standby) = pair();
            let at = Instant::now();
            let ids: Vec<_> = (0..workers)
                .map(|n| leader.0.upsert(facts(&format!("http://w{n}:8000"), 1)).id())
                .collect();
            for rank in 0..8 {
                for id in &ids {
                    leader
                        .0
                        .record(*id, rank, running(3), at, LoadSource::Stream);
                }
            }
            let snapshot = snapshot(&leader, at);
            let mut micros: Vec<u128> = (0..21)
                .map(|_| {
                    let began = Instant::now();
                    follower.restore(&snapshot, at).expect("a restore");
                    began.elapsed().as_micros()
                })
                .collect();
            micros.sort_unstable();
            writeln!(
                out,
                "RESTORE workers={workers} load_rows={} bytes={} median_us={}",
                snapshot.loads.len(),
                prost::Message::encoded_len(&snapshot),
                micros[10]
            )
            .expect("stdout");
            if workers != 512 {
                continue;
            }
            // The tail after the snapshot: 50,000 load rows, then 3,000
            // status changes, each kind in batches of 256.
            let place = follower.place();
            for turn in 0..50_000usize {
                let id = ids[turn % ids.len()];
                leader.0.record(id, 0, running(4), at, LoadSource::Stream);
            }
            let id = ids[0];
            for turn in 1..=3_000 {
                let status = if turn % 2 == 1 {
                    WorkerStatus::Ready
                } else {
                    WorkerStatus::NotReady
                };
                let row = WorkersStore::get(&*leader.0, id).expect("the row");
                leader.0.set_status(id, status, row.revision);
            }
            let sent = entries(&leader.1, place);
            assert_eq!(sent.len(), 53_000, "each write is one event");
            for (kind, events) in [("load", &sent[..50_000]), ("status", &sent[50_000..])] {
                let batches: Vec<_> = events
                    .chunks(256)
                    .map(|chunk| proto::ReplicationBatch {
                        first_seq: chunk[0].at.seq,
                        term: chunk[0].at.term,
                        events: chunk
                            .iter()
                            .map(|entry| worker_wire::replication(entry, at))
                            .collect(),
                        resync_required: false,
                    })
                    .collect();
                let began = Instant::now();
                for batch in &batches {
                    follower.apply(batch, at).expect("a batch that lands");
                }
                let nanos = began.elapsed().as_nanos();
                writeln!(
                    out,
                    "APPLY kind={kind} events={} ns_an_event={}",
                    events.len(),
                    nanos / events.len() as u128
                )
                .expect("stdout");
            }
        }
    }

    const FIVE: Duration = Duration::from_secs(5);

    /// One `Replicate` stream as the scripted leader holds it: the
    /// standby's frames, and where the case puts the batches it answers.
    struct Opened {
        frames: Streaming<proto::ReplicaFrame>,
        batches: mpsc::Sender<Result<proto::ReplicationBatch, Status>>,
    }

    impl Opened {
        /// The standby's next frame; `None` once it left the stream.
        async fn frame(&mut self) -> Option<Frame> {
            let next = timeout(FIVE, self.frames.message()).await;
            let next = next.expect("a frame, or the stream's end, in five seconds");
            next.ok().flatten().and_then(|frame| frame.frame)
        }

        async fn says(&self, batch: proto::ReplicationBatch) {
            let sent = self.batches.send(Ok(batch)).await;
            assert!(sent.is_ok(), "the stream stands");
        }

        async fn refuses(&self, status: Status) {
            let sent = self.batches.send(Err(status)).await;
            assert!(sent.is_ok(), "the stream stands");
        }
    }

    /// A snapshot call, for the case to answer or to leave waiting.
    type Asked = oneshot::Sender<Result<proto::Snapshot, Status>>;

    /// The stream of a call the scripted leader does not serve.
    type Unscripted<T> = ReceiverStream<Result<T, Status>>;

    /// A leader that says what the case tells it: each stream a standby
    /// opens and each snapshot it asks for is handed to the case, with the
    /// name the call stated. While `mute`, it opens no stream.
    #[derive(Clone)]
    struct Scripted {
        streams: mpsc::UnboundedSender<Opened>,
        snapshots: mpsc::UnboundedSender<Asked>,
        names: Arc<Mutex<Vec<String>>>,
        mute: Arc<AtomicBool>,
    }

    impl Scripted {
        fn named<T>(&self, request: &Request<T>) {
            let name = request.metadata().get(ENTRY);
            let name = name.and_then(|name| name.to_str().ok()).unwrap_or_default();
            self.names.lock().push(name.to_owned());
        }
    }

    macro_rules! scripted {
        ($($call:ident($asked:ty) -> $answer:ty;)*) => {
            #[tonic::async_trait]
            impl PolicyController for Scripted {
                type SessionStream = Unscripted<proto::ControllerFrame>;
                type DecideStream = Unscripted<proto::DecideAnswer>;
                type WatchWorkersStream = Unscripted<proto::WorkerEvent>;
                type WatchRegimeStream = Unscripted<proto::Regime>;
                type ReplicateStream = Unscripted<proto::ReplicationBatch>;

                $(async fn $call(
                    &self,
                    _: Request<$asked>,
                ) -> Result<Response<$answer>, Status> {
                    Err(Status::unimplemented("not scripted"))
                })*

                async fn replicate(
                    &self,
                    request: Request<Streaming<proto::ReplicaFrame>>,
                ) -> Result<Response<Self::ReplicateStream>, Status> {
                    self.named(&request);
                    if self.mute.load(Ordering::Relaxed) {
                        pending::<()>().await;
                    }
                    let (batches, sent) = mpsc::channel(16);
                    let frames = request.into_inner();
                    let _ = self.streams.send(Opened { frames, batches });
                    Ok(Response::new(ReceiverStream::new(sent)))
                }

                async fn get_snapshot(
                    &self,
                    request: Request<proto::SnapshotRequest>,
                ) -> Result<Response<proto::Snapshot>, Status> {
                    self.named(&request);
                    let (answer, asked) = oneshot::channel();
                    let _ = self.snapshots.send(answer);
                    let answered = asked.await;
                    answered
                        .unwrap_or_else(|_| Err(Status::unavailable("the case ended")))
                        .map(Response::new)
                }
            }
        };
    }

    scripted! {
        session(Streaming<proto::GatewayFrame>) -> Unscripted<proto::ControllerFrame>;
        decide(Streaming<proto::DecideRequest>) -> Unscripted<proto::DecideAnswer>;
        watch_workers(proto::WatchWorkersRequest) -> Unscripted<proto::WorkerEvent>;
        push_load(Streaming<proto::LoadReport>) -> proto::Ack;
        push_kv_events(Streaming<proto::RelayKvFrame>) -> proto::Ack;
        register_worker(proto::RegisterWorkerRequest) -> proto::WorkerEvent;
        patch_worker(proto::PatchWorkerRequest) -> proto::WorkerEvent;
        deregister_worker(proto::DeregisterWorkerRequest) -> proto::WorkerEvent;
        explain(proto::ExplainRequest) -> proto::DecisionRecord;
        apply_policy_spec(proto::ApplyRequest) -> proto::ApplyResult;
        get_policy_spec(proto::GetPolicySpecRequest) -> proto::PolicySpecDocument;
        rollback_policy_spec(proto::RollbackRequest) -> proto::ApplyResult;
        leader(proto::LeaderRequest) -> proto::LeaderInfo;
        step_down(proto::StepDownRequest) -> proto::Ack;
        watch_regime(proto::RegimeRequest) -> Unscripted<proto::Regime>;
    }

    /// The scripted leader on a loopback port, and what it hands the case.
    struct Bed {
        streams: mpsc::UnboundedReceiver<Opened>,
        snapshots: mpsc::UnboundedReceiver<Asked>,
        names: Arc<Mutex<Vec<String>>>,
        mute: Arc<AtomicBool>,
        addr: SocketAddr,
        _serving: JoinSet<Result<(), Transport>>,
    }

    impl Bed {
        async fn new() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await;
            let listener = listener.expect("a loopback port");
            let addr = listener.local_addr().expect("the bound address");
            let (streams, opened) = mpsc::unbounded_channel();
            let (snapshots, asked) = mpsc::unbounded_channel();
            let names = Arc::new(Mutex::new(Vec::new()));
            let mute = Arc::new(AtomicBool::new(false));
            let leader = Scripted {
                streams,
                snapshots,
                names: Arc::clone(&names),
                mute: Arc::clone(&mute),
            };
            let mut serving = JoinSet::new();
            serving.spawn(
                Server::builder()
                    .add_service(PolicyControllerServer::new(leader))
                    .serve_with_incoming(TcpIncoming::from(listener)),
            );
            Self {
                streams: opened,
                snapshots: asked,
                names,
                mute,
                addr,
                _serving: serving,
            }
        }

        /// The next stream the standby opens.
        async fn opened(&mut self) -> Opened {
            let opened = timeout(FIVE, self.streams.recv()).await;
            opened
                .expect("a stream in five seconds")
                .expect("the leader stands")
        }

        /// The next snapshot the standby asks for.
        async fn asked(&mut self) -> Asked {
            let asked = timeout(FIVE, self.snapshots.recv()).await;
            asked
                .expect("a snapshot call in five seconds")
                .expect("the leader stands")
        }

        /// What the election saw: this leader holds the lease as `id`
        /// under `term`.
        fn holds(&self, id: &str, term: u64) -> Standing {
            Standing {
                holder: format!("{id}@{}", self.addr),
                term,
                looked: true,
            }
        }
    }

    const OWN: &str = "standby-0@127.0.0.1:1";

    /// A replica that follows: its candidate, its election's writer and
    /// its loop, which dials the port of `bed` and holds a stream quiet
    /// for `quiet` at most.
    fn standing_by(bed: &Bed, quiet: Duration) -> (Elector, watch::Sender<Standing>, Standby) {
        let elector = Elector::alone(OWN, ElectionSettings::default())
            .expect("the default settings are in order");
        let (standing, stood) = watch::channel(Standing::default());
        let journal = journaled(&LeasePlace::File("shared".into()));
        let (id, address) = ("standby-0".to_owned(), "127.0.0.1:1".to_owned());
        let catalog = Arc::new(crate::Catalog::builtin());
        let shared = Shared::new(id, address, &elector, stood, journal, catalog, 1)
            .expect("the deciders start");
        let mut standby = Standby::new(Arc::new(shared), bed.addr.port());
        // The bound a replica runs with: three of its leader's heartbeats,
        // so one late beat is not a silent leader.
        assert_eq!(standby.quiet, 3 * BEAT);
        standby.quiet = quiet;
        (elector, standing, standby)
    }

    /// A leader's store with its first worker, under term 1.
    fn led() -> Replica {
        let leader = replica();
        leader.0.set_term(1);
        leader.0.upsert(facts("http://a:8000", 1));
        leader
    }

    fn resync() -> proto::ReplicationBatch {
        proto::ReplicationBatch {
            resync_required: true,
            ..proto::ReplicationBatch::default()
        }
    }

    #[tokio::test]
    async fn the_hello_names_the_standbys_place_and_each_batch_of_events_is_acknowledged_there() {
        let mut bed = Bed::new().await;
        let (_elector, standing, standby) = standing_by(&bed, FIVE);
        // The standby stands past a leader's first writes, under term 1,
        // and the lease names the leader under term 7.
        let leader = led();
        let now = Instant::now();
        follow(&standby.follower, &leader.1, now);
        let place = standby.follower.place();
        assert!(place.seq > 0 && place.term == 1, "{place:?}");
        standing.send_replace(bed.holds("leader-1", 7));
        let named = standby.shared.leader(None);
        let applied = AtomicBool::new(false);
        let script = async {
            let mut opened = bed.opened().await;
            let hello = proto::ReplicaHello {
                replica_id: "standby-0".to_owned(),
                last_applied_seq: place.seq,
                term_seen: 7,
                max_backlog: 0,
                last_applied_term: 1,
            };
            assert_eq!(opened.frame().await, Some(Frame::Hello(hello)));
            // The leader's beat draws no acknowledgement; a batch of
            // events draws one, at the place the standby reached.
            opened.says(proto::ReplicationBatch::default()).await;
            leader.0.upsert(facts("http://b:8000", 1));
            leader.0.upsert(facts("http://c:8000", 1));
            for entry in entries(&leader.1, place) {
                opened.says(batch(&entry, now)).await;
                let seq = entry.at.seq;
                let ack = Frame::Ack(proto::ReplicationAck { seq });
                assert_eq!(opened.frame().await, Some(ack));
            }
            opened.refuses(Status::aborted("deposed")).await;
        };
        let (ended, ()) = tokio::join!(standby.stream(&named, false, &applied), script);
        assert_eq!((ended.next, ended.applied), (Next::Lease, true));
        assert_eq!(ended.why.as_deref(), Some("Aborted: deposed"));
        assert_eq!(standby.follower.place(), leader.1.head(Kinds::All));
        assert_eq!(*bed.names.lock(), ["standby-0"]);
    }

    #[tokio::test]
    async fn a_standby_that_applied_nothing_says_so_and_joins_at_its_leaders_first_write() {
        let mut bed = Bed::new().await;
        let (_elector, standing, standby) = standing_by(&bed, FIVE);
        let leader = led();
        let now = Instant::now();
        standing.send_replace(bed.holds("leader-1", 1));
        let named = standby.shared.leader(None);
        let applied = AtomicBool::new(false);
        let script = async {
            let mut opened = bed.opened().await;
            // No place of any run: zero for both, wherever its log stands.
            let hello = proto::ReplicaHello {
                replica_id: "standby-0".to_owned(),
                last_applied_seq: 0,
                term_seen: 1,
                max_backlog: 0,
                last_applied_term: 0,
            };
            assert_eq!(opened.frame().await, Some(Frame::Hello(hello)));
            // The leader's log from its first write, a term's start at the
            // position after the place the leader's run drew.
            for entry in entries(&leader.1, start(&leader.1)) {
                opened.says(batch(&entry, now)).await;
                let seq = entry.at.seq;
                let ack = Frame::Ack(proto::ReplicationAck { seq });
                assert_eq!(opened.frame().await, Some(ack));
            }
            opened.refuses(Status::aborted("deposed")).await;
        };
        let (ended, ()) = tokio::join!(standby.stream(&named, false, &applied), script);
        assert_eq!((ended.next, ended.applied), (Next::Lease, true));
        assert_eq!(standby.follower.place(), leader.1.head(Kinds::All));
        assert_eq!(start(&standby.shared.feed.log), start(&leader.1));
    }

    #[tokio::test]
    async fn a_resync_is_one_snapshot_and_a_second_with_no_frame_between_waits() {
        let mut bed = Bed::new().await;
        let (_elector, standing, standby) = standing_by(&bed, FIVE);
        let leader = led();
        let now = Instant::now();
        let cut = snapshot(&leader, now);
        // Two ways a join is lost: an event that does not land (the
        // leader's last, past a gap for a standby at the start) and the
        // leader's word.
        let last = entries(&leader.1, start(&leader.1)).pop();
        let gap = batch(&last.expect("the leader wrote"), now);
        standing.send_replace(bed.holds("leader-1", 1));
        let named = standby.shared.leader(None);
        let mut asked = Vec::new();
        for (joined, lost) in [(false, gap), (true, resync())] {
            let applied = AtomicBool::new(false);
            let script = async {
                let opened = bed.opened().await;
                opened.says(lost).await;
                let _ = bed.asked().await.send(Ok(cut.clone()));
                // The stream stands until the standby leaves it.
                opened
            };
            let (ended, _opened) = tokio::join!(standby.stream(&named, joined, &applied), script);
            assert_eq!(standby.follower.place(), leader.1.head(Kinds::All));
            asked.push(standby.pace.due.lock().expect("a snapshot was asked"));
            if joined {
                assert_eq!((ended.next, ended.applied), (Next::Wait, false));
                let why = "restored twice with no frame between";
                assert_eq!(ended.why.as_deref(), Some(why));
            } else {
                assert_eq!((ended.next, ended.applied), (Next::Join, true));
                let lag = standby.shared.lag.age(Instant::now());
                assert!(lag.is_some(), "a restore is a frame applied");
            }
        }
        // The second snapshot waited out the beat since the first was asked.
        assert!(asked[1] >= asked[0] + RESYNC, "{asked:?}");
        // Every call stated the replica's name: two streams, two snapshots.
        assert_eq!(*bed.names.lock(), ["standby-0"; 4]);
    }

    #[tokio::test]
    async fn a_replica_that_leads_by_now_takes_no_snapshot() {
        let mut bed = Bed::new().await;
        let (mut elector, standing, standby) = standing_by(&bed, FIVE);
        standing.send_replace(bed.holds("leader-1", 1));
        let named = standby.shared.leader(None);
        let applied = AtomicBool::new(false);
        let script = async {
            let opened = bed.opened().await;
            // The replica takes its own lease while the stream stands.
            let took = elector.tick(Instant::now());
            assert!(matches!(took, Ok(Tick::Acquired { .. })), "{took:?}");
            opened.says(resync()).await;
            opened
        };
        let (ended, _opened) = tokio::join!(standby.stream(&named, false, &applied), script);
        assert_eq!((ended.next, ended.applied), (Next::Lease, false));
        assert!(ended.why.is_none());
        assert!(bed.snapshots.try_recv().is_err(), "no snapshot was asked");
    }

    #[tokio::test]
    async fn the_loop_leaves_a_stream_once_the_lease_names_another_leader_or_this_replica() {
        let mut bed = Bed::new().await;
        let (mut elector, standing, standby) = standing_by(&bed, FIVE);
        let (stop, stopped) = watch::channel(false);
        standing.send_replace(bed.holds("leader-1", 1));
        let mut running = JoinSet::new();
        running.spawn(standby.run(stopped));
        let seen = |frame: Option<Frame>| match frame {
            Some(Frame::Hello(hello)) => Some(hello.term_seen),
            _ => None,
        };
        let mut first = bed.opened().await;
        assert_eq!(seen(first.frame().await), Some(1));
        // The lease names another leader. The first one's stream stands
        // and says nothing: it is left, and the leader named is joined.
        standing.send_replace(bed.holds("leader-2", 2));
        let mut second = bed.opened().await;
        assert_eq!(seen(second.frame().await), Some(2));
        assert_eq!(first.frame().await, None, "the first stream is left");
        // The same leader under its next term, the lease lost and taken
        // again: the stream of its last term is left too.
        standing.send_replace(bed.holds("leader-2", 3));
        let mut third = bed.opened().await;
        assert_eq!(seen(third.frame().await), Some(3));
        assert_eq!(second.frame().await, None, "the last term's stream");
        // This replica takes the lease: it leaves the stream, and at the
        // election's next step it dials nobody.
        let took = elector.tick(Instant::now());
        assert!(matches!(took, Ok(Tick::Acquired { .. })), "{took:?}");
        let own = Standing {
            holder: OWN.to_owned(),
            term: 4,
            looked: true,
        };
        standing.send_replace(own.clone());
        assert_eq!(third.frame().await, None, "a leader follows nobody");
        standing.send_replace(own);
        let dialed = timeout(Duration::from_millis(300), bed.streams.recv()).await;
        assert!(dialed.is_err(), "a leader dials nobody");
        stop.send_replace(true);
        let ended = timeout(FIVE, running.join_next()).await;
        assert!(matches!(ended, Ok(Some(Ok(())))), "the stop ends the loop");
    }

    #[tokio::test]
    async fn a_leader_that_says_nothing_is_left_at_the_quiet_bound() {
        let mut bed = Bed::new().await;
        let (_elector, standing, standby) = standing_by(&bed, Duration::from_millis(100));
        standing.send_replace(bed.holds("leader-1", 1));
        let named = standby.shared.leader(None);
        // A stream that stands and brings no frame after its first.
        let applied = AtomicBool::new(false);
        let script = async {
            let opened = bed.opened().await;
            opened.says(proto::ReplicationBatch::default()).await;
            opened
        };
        let left = async { tokio::join!(standby.stream(&named, false, &applied), script) };
        let (ended, opened) = timeout(FIVE, left).await.expect("the stream is left");
        assert_eq!((ended.next, ended.applied), (Next::Wait, true));
        let why = "the leader sent no frame for 100ms";
        assert_eq!(ended.why.as_deref(), Some(why));
        drop(opened);
        // A snapshot that does not arrive: ten quiet bounds.
        let applied = AtomicBool::new(false);
        let script = async {
            let opened = bed.opened().await;
            opened.says(resync()).await;
            (opened, bed.asked().await)
        };
        let left = async { tokio::join!(standby.stream(&named, false, &applied), script) };
        let (ended, held) = timeout(FIVE, left).await.expect("the stream is left");
        assert_eq!((ended.next, ended.applied), (Next::Wait, false));
        let why = "the leader's snapshot did not arrive in 1s";
        assert_eq!(ended.why.as_deref(), Some(why));
        drop(held);
        // A leader that opens no stream.
        bed.mute.store(true, Ordering::Relaxed);
        let applied = AtomicBool::new(false);
        let left = standby.stream(&named, false, &applied);
        let ended = timeout(FIVE, left).await.expect("the stream is left");
        assert_eq!((ended.next, ended.applied), (Next::Wait, false));
        let why = "the leader opened no stream for 100ms";
        assert_eq!(ended.why.as_deref(), Some(why));
    }

    /// The lines a case's subscriber kept.
    #[derive(Clone, Default)]
    struct Logged(Arc<Mutex<Vec<u8>>>);

    impl io::Write for Logged {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn a_streak_of_streams_that_applied_nothing_is_said_once_and_again_after_one_that_did() {
        let lines = Logged::default();
        let writer = lines.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        // A second subscriber stays registered beside it: with one alone a
        // line is on or off by the thread that reaches it first.
        let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
        let _kept = tracing::subscriber::set_default(subscriber);
        let said = || {
            let kept = lines.0.lock();
            let text = String::from_utf8_lossy(&kept);
            text.matches("this standby does not follow its leader")
                .count()
        };
        let mut bed = Bed::new().await;
        let (_elector, standing, standby) = standing_by(&bed, FIVE);
        let (stop, stopped) = watch::channel(false);
        standing.send_replace(bed.holds("leader-1", 1));
        let mut running = JoinSet::new();
        running.spawn(standby.run(stopped));
        // Three streams the leader refuses: one line.
        for _ in 0..3 {
            let opened = bed.opened().await;
            opened.refuses(Status::unavailable("not yet")).await;
        }
        // A stream that applies a frame ends the streak: the reason of the
        // next one is said anew.
        let opened = bed.opened().await;
        assert_eq!(said(), 1);
        opened.says(proto::ReplicationBatch::default()).await;
        opened.refuses(Status::unavailable("not yet")).await;
        let _next = bed.opened().await;
        assert_eq!(said(), 2);
        stop.send_replace(true);
        let ended = timeout(FIVE, running.join_next()).await;
        assert!(matches!(ended, Ok(Some(Ok(())))), "the stop ends the loop");
    }
}
