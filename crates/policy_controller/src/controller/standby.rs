//! A replica that does not lead follows (design 5.4): the standby's half of
//! the replication scope. It applies the batches of `Replicate` and, where
//! the leader's log does not continue its own, is restored from the
//! leader's snapshot, which arrives in parts on the same stream.
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
    store::{Cut, Kinds, LeasesStore, LoadSample, MemoryStore, Position, WorkerRow},
    view::{fingerprint, WorkerId},
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

    /// The place a standby that must be restored names in its hello: its
    /// head's number under no term. No log holds a position that names no
    /// term, so its leader answers the snapshot, whatever the leader's log
    /// holds at that number. The number is past zero: zero under no term
    /// is the place of a standby that applied nothing.
    pub(crate) fn asking(&self) -> Position {
        Position {
            seq: self.log.head(Kinds::All).seq.max(1),
            term: 0,
        }
    }

    /// The store restored from the leader's snapshot, its parts staged
    /// whole ([`Staged`]): the standby stands at the cut's place after it.
    /// `false` where the log refuses that place (a history that is not
    /// ahead of its own), the store as it was.
    pub(crate) fn restore(&self, cut: Cut) -> bool {
        self.store.restore_from(cut)
    }

    /// One batch applied at `now`: each event at its position under the
    /// batch's term, by the store's door for a followed write of its kind,
    /// which alone lands there: a write of the store's own that comes
    /// between takes no part in it, and the event does not land. A load row
    /// stands with the instant of its apply. The first event that does not
    /// land stops the batch; the events before it stand, each at its place.
    /// A standby that applied nothing takes its leader's first position, a
    /// term's start, wherever the leader's run drew it ([`Log::join`]); no
    /// other event lands on it. A batch that carries a part of a snapshot
    /// is refused by its path: a part is read inside the snapshot its
    /// leader announced alone ([`Staged::take`]).
    pub(crate) fn apply(&self, batch: &proto::ReplicationBatch, now: Instant) -> Result<(), Halt> {
        // No leader of this build sends a part with no word before it: the
        // wire's guard alone.
        if batch.snapshot.is_some() {
            return Err(Halt::Unread(Refused {
                path: "batch.snapshot",
                why: "a part of a snapshot the leader did not announce",
            }));
        }
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
                    self.store.follow_term(at);
                    continue;
                }
                Err(Lost) => return Err(Halt::Lost(at)),
            };
            match &event.event {
                Some(Sent::Membership(told)) => {
                    let (epoch, change, row) =
                        worker_wire::told_change(told, now).map_err(Halt::Unread)?;
                    self.store.follow(at, epoch, change, row);
                }
                Some(Sent::Load(told)) => {
                    let (id, rank, row, source) =
                        worker_wire::told_load(told).map_err(Halt::Unread)?;
                    self.store.follow_load(at, id, rank, row, now, source);
                }
                Some(Sent::TermStart(_)) => self.store.follow_term(at),
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

/// A leader's snapshot as its parts arrive: the rows and the load rows,
/// held apart from the store until the last part. The store is restored
/// from all of them or from none: a stream that ends inside a snapshot
/// leaves it as it was.
#[derive(Debug, Default)]
pub(crate) struct Staged {
    /// The snapshot's place and membership epoch, as its first part said
    /// them: every part says the same.
    at: Option<(Position, u64)>,
    /// The next part's number.
    part: u32,
    rows: Vec<WorkerRow>,
    loads: Vec<(WorkerId, u16, LoadSample)>,
}

impl Staged {
    /// One batch of the stream inside a snapshot, read at `now`: the
    /// snapshot's next part, staged, and at the last part the cut the
    /// store is restored from. A part is its snapshot's next where its
    /// batch carries no event and says the place and the epoch the first
    /// part said, under the next number. Anything else is refused by its
    /// path, and the stream is left with nothing restored.
    pub(crate) fn take(
        &mut self,
        batch: &proto::ReplicationBatch,
        now: Instant,
    ) -> Result<Option<Cut>, Refused> {
        let refused = |path, why| Err(Refused { path, why });
        let Some(part) = &batch.snapshot else {
            return refused(
                "batch.snapshot",
                "no part of the snapshot the leader announced",
            );
        };
        if !batch.events.is_empty() {
            return refused("batch.events", "events in a part of a snapshot");
        }
        // A part says its snapshot's place as a heartbeat says the head.
        let Some(seq) = batch.first_seq.checked_sub(1) else {
            return refused("batch.first_seq", "a snapshot at no place");
        };
        let at = Position {
            seq,
            term: batch.term,
        };
        if *self.at.get_or_insert((at, part.membership_epoch)) != (at, part.membership_epoch) {
            return refused(
                "batch.first_seq",
                "a part at another place than its snapshot's",
            );
        }
        if part.part != self.part {
            return refused("batch.snapshot.part", "a part out of its snapshot's order");
        }
        self.part += 1;
        worker_wire::told_part(part, now, &mut self.rows, &mut self.loads)?;
        Ok(part.last.then(|| Cut {
            at,
            epoch: part.membership_epoch,
            rows: std::mem::take(&mut self.rows),
            loads: std::mem::take(&mut self.loads),
        }))
    }
}

/// How far behind its leader a standby is, as its last frame at the head
/// says it: a frame that left it at the head its leader stated there, or
/// any applied frame of a leader that states none. The loop writes it, and
/// `Leader` and the readiness read it.
#[derive(Debug)]
pub(crate) struct Lag {
    origin: Instant,
    /// The lag this replica reads synced under ([`synced_within`]).
    within: Duration,
    /// The last frame at the head, in microseconds past the origin plus one;
    /// `0` before any, and once this replica has led since.
    frame: AtomicU64,
    /// Whether the serve runs a standby that writes this lag.
    follows: AtomicBool,
    /// How many times the standby asked its leader to be restored.
    asked: AtomicU64,
    /// The turns a reader may have missed, which looks again at each: a
    /// frame written after no frame, or when the frame before was as old
    /// as the bound by the clock of the write, and a frame forgotten.
    turns: watch::Sender<u64>,
}

impl Lag {
    pub(crate) fn new(within: Duration) -> Self {
        Self {
            origin: Instant::now(),
            within,
            frame: AtomicU64::new(0),
            follows: AtomicBool::new(false),
            asked: AtomicU64::new(0),
            turns: watch::Sender::new(0),
        }
    }

    /// A frame of the leader's stream, or the last part of its snapshot,
    /// left this replica at the head at `at`.
    pub(crate) fn applied(&self, at: Instant) {
        let before = self.store(at);
        // The clock is read after the write: whoever read the frame before
        // this one read it no later.
        self.turn(before, Instant::now());
    }

    /// Forgets the last frame at the head, and tells a turn where there was
    /// one. The standby loop's call once this replica has held the lease:
    /// a frame it applied before its own lead says nothing of the leader
    /// it follows next.
    pub(crate) fn forget(&self) {
        if self.frame.swap(0, Ordering::Relaxed) != 0 {
            self.turns.send_modify(|turns| *turns += 1);
        }
    }

    /// Stores the frame's instant, and returns what the frame before left.
    fn store(&self, at: Instant) -> u64 {
        let micros = at.saturating_duration_since(self.origin).as_micros();
        let micros = u64::try_from(micros).unwrap_or(u64::MAX - 1);
        self.frame.swap(micros + 1, Ordering::Relaxed)
    }

    /// Tells a turn where the frame `before` left was none, or was as old
    /// as the bound at `wrote`, the clock of the write that replaced it. A
    /// reader judges the sync by the clock at its read, so one may have
    /// read it gone whatever instant the new frame carries: an apply that
    /// straddles the lapse writes a frame dated before it.
    fn turn(&self, before: u64, wrote: Instant) {
        let silent = before.checked_sub(1).is_none_or(|before| {
            let last = self.origin + Duration::from_micros(before);
            wrote.saturating_duration_since(last) >= self.within
        });
        if silent {
            self.turns.send_modify(|turns| *turns += 1);
        }
    }

    /// The last frame at the head at `now`, by one read of it: its age, and
    /// the instant the sync lapses with no further such frame, none where the
    /// frame is as old as the bound already. `None` where the lag holds no
    /// frame. This replica is synced while there is such an instant: the
    /// `Leader` call's `synced` and a follower's readiness are this read.
    pub(crate) fn read(&self, now: Instant) -> Option<(Duration, Option<Instant>)> {
        let frame = self.frame.load(Ordering::Relaxed).checked_sub(1)?;
        let applied = self.origin + Duration::from_micros(frame);
        let lapse = applied + self.within;
        let age = now.saturating_duration_since(applied);
        Some((age, (now < lapse).then_some(lapse)))
    }

    /// The age at `now` of the last frame at the head; `None` where the lag
    /// holds none.
    #[cfg(test)]
    pub(crate) fn age(&self, now: Instant) -> Option<Duration> {
        self.read(now).map(|(age, _)| age)
    }

    /// Whether this replica is synced at `now`: its last frame at the head
    /// is younger than the bound.
    #[cfg(test)]
    pub(crate) fn synced(&self, now: Instant) -> bool {
        self.read(now).is_some_and(|(_, lapse)| lapse.is_some())
    }

    /// The bound [`Lag::synced`] reads under.
    #[cfg(test)]
    pub(crate) fn within(&self) -> Duration {
        self.within
    }

    /// The standby asks its leader to be restored: an event of the
    /// leader's did not land on it.
    pub(crate) fn asked(&self) {
        self.asked.fetch_add(1, Ordering::Relaxed);
    }

    /// How many times this replica asked to be restored since its start.
    pub(crate) fn asks(&self) -> u64 {
        self.asked.load(Ordering::Relaxed)
    }

    /// The serve runs a standby from here: this replica is ready as a
    /// follower only while it is synced.
    pub(crate) fn follow(&self) {
        self.follows.store(true, Ordering::Relaxed);
    }

    /// Whether the serve runs a standby that writes this lag.
    pub(crate) fn follows(&self) -> bool {
        self.follows.load(Ordering::Relaxed)
    }

    /// A reader's view of the turns: it changes when a frame ends a
    /// silence that had ended the sync.
    pub(crate) fn turns(&self) -> watch::Receiver<u64> {
        self.turns.subscribe()
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

/// How often a standby asks its leader to be restored at most. A cut holds
/// the leader's writers for its length and nothing else paces the ask: a
/// standby on which its leader's events keep not landing asks one a beat.
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
    /// An event did not land: the next stream opens at once, and asks to
    /// be restored.
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
    /// When this replica may next ask to be restored ([`RESYNC`]).
    pace: Pace,
    /// Whether the next hello asks to be restored: an event of the leader's
    /// did not land, and no snapshot was restored since.
    asks: AtomicBool,
    /// Whether no event landed since the last restore.
    fresh: AtomicBool,
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
            asks: AtomicBool::new(false),
            fresh: AtomicBool::new(false),
        }
    }

    /// Follows until `stop` reads `true`: one stream after another, each
    /// against the holder the lease names at that moment, and none while
    /// this replica leads or nobody is named. A streak of streams that
    /// applied nothing is said once, by its reason, and waits longer a
    /// stream ([`backoff`]); the lease naming another holder ends the wait,
    /// and the stream wherever it stands, the leader's answer not waited
    /// for. A replica that has held the lease is synced again at the first
    /// frame of the leader it follows next ([`Lag::forget`]).
    pub(crate) async fn run(self, mut stop: watch::Receiver<bool>) {
        let mut steps = self.shared.standing.clone();
        let mut streak = Streak::default();
        let mut led = self.shared.fence.term();
        loop {
            let leading = self.shared.fence.leader_term(Instant::now());
            let leader = self.shared.leader(leading);
            // The fence's term moves at a take alone: this replica has
            // held the lease since the loop's last look.
            let term = self.shared.fence.term();
            if term != led {
                self.shared.lag.forget();
                led = term;
            }
            let next = if leader.is_leader || leader.address.is_empty() {
                Next::Lease
            } else {
                let applied = AtomicBool::new(false);
                let ended = tokio::select! {
                    ended = self.stream(&leader, &applied) => ended,
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

    /// A stream whose snapshot this replica's log refused, at `at`: the
    /// lease is read again.
    fn unfollowed(at: Position, applied: bool) -> Self {
        Self {
            next: Next::Lease,
            applied,
            why: Some(format!(
                "the leader's snapshot stands at {} under term {}, which this replica's log does not follow",
                at.seq, at.term
            )),
        }
    }
}

impl Standby {
    /// One stream's life against `leader`: the dial, the hello at the log's
    /// head, then every batch applied and acknowledged until the stream
    /// ends. Where the leader's log does not continue this one the leader
    /// says so (`resync_required`) and sends its snapshot on the same
    /// stream, in parts: they are staged, the store is restored at the last
    /// one, whole, and the stream goes on from the snapshot's place. A
    /// stream that ends inside a snapshot leaves the store as it was.
    ///
    /// An event that does not land ends the stream, and the next one asks
    /// to be restored ([`Standby::lost`]). `applied` is set once a frame of
    /// the stream landed: the loop reads it where it leaves the stream
    /// itself. A leader that opens no stream, or sends no frame, within the
    /// quiet bound is left and asked again.
    async fn stream(&self, leader: &Leader, applied: &AtomicBool) -> Ended {
        let asks = self.asks.load(Ordering::Relaxed);
        if asks {
            self.pace.wait().await;
            if self.shared.fence.leader_term(Instant::now()).is_some() {
                return Ended::leads(false);
            }
        }
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
        let place = if asks {
            self.follower.asking()
        } else {
            self.follower.place()
        };
        let hello = proto::ReplicaHello {
            replica_id: self.shared.controller_id.clone(),
            last_applied_seq: place.seq,
            // The floor: the lease's term as this replica reads it, never
            // under the term its store stands under. A hello that asks to
            // be restored names its place under no term, and states the
            // floor all the same.
            term_seen: leader.term.max(self.shared.store.term()),
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
        // The leader's snapshot while its parts arrive.
        let mut staged: Option<Staged> = None;
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
            if batch.resync_required {
                // The leader's snapshot follows: a second word starts its
                // staging over.
                staged = Some(Staged::default());
                continue;
            }
            if let Some(staging) = staged.as_mut() {
                let cut = match staging.take(&batch, now) {
                    Ok(Some(cut)) => cut,
                    Ok(None) => continue,
                    Err(refused) => return Ended::waits(landed, unread(refused)),
                };
                staged = None;
                let (at, rows) = (cut.at, cut.rows.len());
                if !self.follower.restore(cut) {
                    return Ended::unfollowed(at, landed);
                }
                info!(
                    controller_id = %self.shared.controller_id,
                    seq = at.seq,
                    term = at.term,
                    rows,
                    "restored from the leader's snapshot"
                );
                self.asks.store(false, Ordering::Relaxed);
                self.fresh.store(true, Ordering::Relaxed);
            } else {
                match self.follower.apply(&batch, now) {
                    Ok(()) => {}
                    Err(Halt::Lost(at)) => return self.lost(leader, at, landed),
                    Err(Halt::Unread(refused)) => return Ended::waits(landed, unread(refused)),
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
                if !batch.events.is_empty() {
                    self.fresh.store(false, Ordering::Relaxed);
                    // An acknowledgement names the place reached: one the
                    // channel has no room for is covered by the next.
                    let seq = self.follower.place().seq;
                    let _ = frames.try_send(frame(Frame::Ack(proto::ReplicationAck { seq })));
                }
            }
            applied.store(true, Ordering::Relaxed);
            // The lag's instant is the last frame that left this replica at
            // the head its leader stated: a frame inside a replay, or in
            // the tail a restore leaves, moves nothing. A frame that states
            // no head, a leader of a build before the field, moves it as
            // every applied frame did.
            if batch.head_seq == 0 || batch.head_seq == self.follower.place().seq {
                self.shared.lag.applied(now);
            }
        }
    }

    /// An event of `leader`'s did not land at `at`: the standby's tables or
    /// its log refused it, though the leader's log may hold its place. The
    /// next stream asks to be restored ([`Follower::asking`]), and the ask
    /// is counted and said here: the leader cannot tell it from a head
    /// under no term. The next stream opens at once, unless this event
    /// follows a restore with no event landed between: two in a row are
    /// two restores a streak's wait apart, never a loop at the stream's
    /// speed.
    fn lost(&self, leader: &Leader, at: Position, landed: bool) -> Ended {
        self.asks.store(true, Ordering::Relaxed);
        self.shared.lag.asked();
        warn!(
            controller_id = %self.shared.controller_id,
            leader = %leader.controller_id,
            seq = at.seq,
            term = at.term,
            "an event of the leader's did not land on this standby; it asks to be restored from a snapshot"
        );
        if self.fresh.load(Ordering::Relaxed) {
            let why = "restored twice with no event landed between".to_owned();
            return Ended::waits(false, why);
        }
        Ended {
            next: Next::Join,
            applied: landed,
            why: None,
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
    use tokio::{net::TcpListener, task::JoinSet};
    use tonic::{
        transport::{server::TcpIncoming, Error as Transport, Server},
        Response, Streaming,
    };

    use super::*;
    use crate::{
        controller::{
            election::LeasePlace,
            replication::{journaled, journaled_from, parts, Replication},
        },
        elector::{ElectionSettings, Elector, Tick},
        proto::policy_controller_server::{PolicyController, PolicyControllerServer},
        replication::{Entry, Event},
        store::{LoadSink, LoadSource, WorkerFacts, WorkerStatus, WorkersStore},
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

    /// One entry as the batch a leader sends it in at `now`, the last its
    /// log holds.
    fn batch(entry: &Entry, now: Instant) -> proto::ReplicationBatch {
        batch_of(std::slice::from_ref(entry), entry.at.seq, now)
    }

    /// `entries`, written under one term, as the batch a leader sends them
    /// in at `now` while its log's head stands at `head`.
    fn batch_of(entries: &[Entry], head: u64, now: Instant) -> proto::ReplicationBatch {
        proto::ReplicationBatch {
            first_seq: entries[0].at.seq,
            term: entries[0].at.term,
            events: entries
                .iter()
                .map(|entry| worker_wire::replication(entry, now))
                .collect(),
            resync_required: false,
            snapshot: None,
            head_seq: head,
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

    /// The parts of the leader's snapshot at `at`, as its replication scope
    /// sends them under the term its store wrote.
    fn snapshot((store, log): &Replica, at: Instant) -> Vec<proto::ReplicationBatch> {
        let mut elector =
            Elector::alone("controller-0@10.0.0.7:30100", ElectionSettings::default())
                .expect("the default settings are in order");
        let Ok(Tick::Acquired { term }) = elector.tick(at) else {
            panic!("a free lease is taken at the first tick");
        };
        let cut = Replication::new(Arc::clone(store), Arc::clone(log), elector.fence())
            .cut(term)
            .expect("a cut");
        parts(cut, at)
    }

    /// The cut a standby stages from the `parts` of a snapshot, read at
    /// `at`.
    fn staged(parts: &[proto::ReplicationBatch], at: Instant) -> Cut {
        let mut staged = Staged::default();
        let mut cut = None;
        for part in parts {
            assert!(cut.is_none(), "a part after the last");
            cut = staged.take(part, at).expect("the snapshot's next part");
        }
        cut.expect("the last part")
    }

    #[test]
    fn a_standby_restored_from_the_snapshot_continues_from_its_place() {
        let (leader, follower, (standby, _followed)) = pair();
        let at = Instant::now();
        first(&leader.0, at);
        // The standby held a row of its own, under no term.
        standby.upsert(facts("http://stale:8000", 1));
        let first = snapshot(&leader, at);
        assert!(follower.restore(staged(&first, at)));
        assert_eq!(follower.place(), leader.1.head(Kinds::All));
        assert_eq!(held(&standby), held(&leader.0));
        rest(&leader.0, at);
        follow(&follower, &leader.1, at);
        assert_eq!(held(&standby), held(&leader.0));
        // A snapshot of the term it followed before is a history it left:
        // refused, the store as it was.
        let place = follower.place();
        assert!(!follower.restore(staged(&first, at)));
        assert_eq!(follower.place(), place);
        assert_eq!(held(&standby), held(&leader.0));
    }

    #[test]
    fn a_batch_that_is_not_its_snapshots_next_part_is_refused_by_its_path() {
        let (leader, ..) = pair();
        let at = Instant::now();
        first(&leader.0, at);
        let parts = snapshot(&leader, at);
        assert_eq!(parts.len(), 2, "the rows' part and the load rows'");
        let events = batch(&entries(&leader.1, start(&leader.1)).remove(0), at);
        let next = parts[1].clone();
        let part = next.snapshot.clone().expect("a part");
        // Each after the snapshot's first part, which is staged.
        let refused = |batch: proto::ReplicationBatch| {
            let mut staged = Staged::default();
            assert!(matches!(staged.take(&parts[0], at), Ok(None)));
            let refused = staged.take(&batch, at).expect_err("not the next part");
            (refused.path, refused.why)
        };
        let none = (
            "batch.snapshot",
            "no part of the snapshot the leader announced",
        );
        assert_eq!(refused(events.clone()), none);
        assert_eq!(refused(proto::ReplicationBatch::default()), none);
        let carried = proto::ReplicationBatch {
            events: events.events,
            ..next.clone()
        };
        let why = "events in a part of a snapshot";
        assert_eq!(refused(carried), ("batch.events", why));
        let why = "a part out of its snapshot's order";
        assert_eq!(refused(parts[0].clone()), ("batch.snapshot.part", why));
        let elsewhere = [
            proto::ReplicationBatch {
                first_seq: next.first_seq + 1,
                ..next.clone()
            },
            proto::ReplicationBatch {
                term: next.term + 1,
                ..next.clone()
            },
            proto::ReplicationBatch {
                snapshot: Some(proto::ReplicationSnapshot {
                    membership_epoch: part.membership_epoch + 1,
                    ..part
                }),
                ..next.clone()
            },
        ];
        let why = "a part at another place than its snapshot's";
        for batch in elsewhere {
            assert_eq!(refused(batch), ("batch.first_seq", why));
        }
        // A first part that says position zero stands at no place.
        let nowhere = proto::ReplicationBatch {
            first_seq: 0,
            ..parts[0].clone()
        };
        let refused = Staged::default().take(&nowhere, at).expect_err("no place");
        assert_eq!(
            (refused.path, refused.why),
            ("batch.first_seq", "a snapshot at no place")
        );
    }

    /// A part the leader did not announce is no heartbeat and no batch of
    /// events: the applier refuses it by its path with nothing applied,
    /// the place and the tables as they were.
    #[test]
    fn a_part_the_leader_did_not_announce_is_refused_by_its_path() {
        let (leader, follower, (standby, _followed)) = pair();
        let at = Instant::now();
        first(&leader.0, at);
        let parts = snapshot(&leader, at);
        let (place, before) = (follower.place(), held(&standby));
        let why = "a part of a snapshot the leader did not announce";
        for part in &parts {
            let Err(Halt::Unread(refused)) = follower.apply(part, at) else {
                panic!("a part with no word before it was applied");
            };
            assert_eq!((refused.path, refused.why), ("batch.snapshot", why));
        }
        assert_eq!((follower.place(), held(&standby)), (place, before));
    }

    #[test]
    fn a_standby_that_asks_names_its_heads_number_past_zero_under_no_term() {
        // A log whose places start at the start: its head is zero under no
        // term, the place of a standby that applied nothing and no ask.
        let (store, feed) = journaled_from(&LeasePlace::File("shared".into()), 0);
        let follower = Follower::new(store, Arc::clone(&feed.log));
        assert_eq!(follower.place(), Position::default());
        assert_eq!(follower.asking(), Position { seq: 1, term: 0 });
        // Past it, the head's own number.
        let leader = led();
        follow(&follower, &leader.1, Instant::now());
        let head = follower.place();
        assert!(head.seq > 1 && head.term == 1, "{head:?}");
        assert_eq!(follower.asking(), Position { term: 0, ..head });
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
        let lag = Lag::new(Duration::from_secs(1));
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
    fn the_sync_holds_under_the_bound_and_turns_by_the_clock_of_the_write() {
        let bound = Duration::from_secs(1);
        let lag = Lag::new(bound);
        assert_eq!(lag.within(), bound);
        let turns = lag.turns();
        let turned = || *turns.borrow();
        // A frame dated `at`, written when the clock reads `wrote`.
        let write = |at, wrote| lag.turn(lag.store(at), wrote);
        let start = Instant::now();
        // Before any frame: not synced, nothing to lapse, no turn told.
        assert!(!lag.synced(start));
        assert_eq!((lag.read(start), turned()), (None, 0));

        // The first frame turns the sync.
        write(start, start);
        // The lag holds the frame's instant at a whole microsecond.
        let frame = start - lag.age(start).expect("a frame applied");
        assert_eq!(turned(), 1);
        let under = bound - Duration::from_micros(1);
        assert!(lag.synced(frame + under));
        assert_eq!(lag.read(frame + under), Some((under, Some(frame + bound))));
        // At the bound it is gone, and there is nothing left to lapse.
        assert!(!lag.synced(frame + bound));
        assert_eq!(lag.read(frame + bound), Some((bound, None)));

        // A frame written inside the bound keeps the sync and tells no
        // turn.
        let last = frame + under;
        write(last, last);
        assert_eq!(turned(), 1);
        // An apply that straddles the lapse: its frame is dated inside the
        // bound and written at it, after a reader read the sync gone.
        write(last + under, last + bound);
        assert_eq!(turned(), 2);
        assert!(lag.synced(last + bound));

        // A replica follows from the serve's word alone.
        assert!(!lag.follows());
        lag.follow();
        assert!(lag.follows());
    }

    #[test]
    fn an_applied_frame_dated_before_the_lapse_and_written_after_it_tells_a_turn() {
        // The loop reads its instant, applies the batch, then writes the
        // lag: the clock of that write says whether a reader saw the sync
        // gone, not the instant the frame carries.
        let bound = Duration::from_millis(50);
        let lag = Lag::new(bound);
        let turns = lag.turns();
        let first = Instant::now();
        lag.applied(first);
        assert_eq!(*turns.borrow(), 1);
        // The bound passes with no frame.
        std::thread::sleep(bound + Duration::from_millis(10));
        // A frame dated inside the bound, written now.
        lag.applied(first + bound - Duration::from_millis(1));
        assert_eq!(*turns.borrow(), 2);
    }

    #[test]
    fn a_forgotten_frame_reads_as_none_and_tells_a_turn() {
        let lag = Lag::new(Duration::from_secs(1));
        let turns = lag.turns();
        let now = Instant::now();
        // Nothing to forget: no turn told.
        lag.forget();
        assert_eq!(*turns.borrow(), 0);
        lag.applied(now);
        assert!(lag.synced(now));
        // The frame is gone, and a reader that read it looks again.
        lag.forget();
        assert_eq!((lag.read(now), *turns.borrow()), (None, 2));
        lag.forget();
        assert_eq!(*turns.borrow(), 2);
        // The next frame is a first frame.
        lag.applied(now);
        assert_eq!(*turns.borrow(), 3);
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
            (Status::resource_exhausted("over its budget"), Next::Wait),
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
    /// restore from the snapshot's parts costs at 512 and 4,096 workers of
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
            let parts = snapshot(&leader, at);
            let mut micros: Vec<u128> = (0..21)
                .map(|_| {
                    let began = Instant::now();
                    assert!(follower.restore(staged(&parts, at)), "a restore");
                    began.elapsed().as_micros()
                })
                .collect();
            micros.sort_unstable();
            let told = parts.iter().filter_map(|part| part.snapshot.as_ref());
            let load_rows: usize = told.map(|part| part.loads.len()).sum();
            let bytes: usize = parts.iter().map(prost::Message::encoded_len).sum();
            writeln!(
                out,
                "RESTORE workers={workers} load_rows={load_rows} parts={} bytes={bytes} median_us={}",
                parts.len(),
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
                        snapshot: None,
                        head_seq: 0,
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

    /// The stream of a call the scripted leader does not serve.
    type Unscripted<T> = ReceiverStream<Result<T, Status>>;

    /// A leader that says what the case tells it: each stream a standby
    /// opens is handed to the case, with the name the call stated. While
    /// `mute`, it opens no stream.
    #[derive(Clone)]
    struct Scripted {
        streams: mpsc::UnboundedSender<Opened>,
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
        get_snapshot(proto::SnapshotRequest) -> proto::Snapshot;
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
            let names = Arc::new(Mutex::new(Vec::new()));
            let mute = Arc::new(AtomicBool::new(false));
            let leader = Scripted {
                streams,
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
        let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
        assert_eq!((ended.next, ended.applied), (Next::Lease, true));
        assert_eq!(ended.why.as_deref(), Some("Aborted: deposed"));
        assert_eq!(standby.follower.place(), leader.1.head(Kinds::All));
        assert_eq!(*bed.names.lock(), ["standby-0"]);
    }

    #[tokio::test]
    async fn a_standby_reads_synced_at_the_frame_that_leaves_it_at_its_leaders_head() {
        // A leader that states its head, and one of a build before the
        // field, which states none.
        for stated in [true, false] {
            let mut bed = Bed::new().await;
            let (_elector, standing, standby) = standing_by(&bed, FIVE);
            // The standby stands past a leader's first writes, and the
            // leader wrote three rows since: a replay of two frames.
            let leader = led();
            let now = Instant::now();
            follow(&standby.follower, &leader.1, now);
            let place = standby.follower.place();
            for key in ["http://b:8000", "http://c:8000", "http://d:8000"] {
                leader.0.upsert(facts(key, 1));
            }
            let tail = entries(&leader.1, place);
            let head = if stated {
                leader.1.head(Kinds::All).seq
            } else {
                0
            };
            standing.send_replace(bed.holds("leader-1", 1));
            let named = standby.shared.leader(None);
            let applied = AtomicBool::new(false);
            let synced = || standby.shared.lag.age(Instant::now()).is_some();
            let script = async {
                let mut opened = bed.opened().await;
                assert!(matches!(opened.frame().await, Some(Frame::Hello(_))));
                assert!(!synced(), "before any frame");
                let ack = |entry: &Entry| {
                    let seq = entry.at.seq;
                    Some(Frame::Ack(proto::ReplicationAck { seq }))
                };
                // The replay's first frame ends short of the head it
                // states: the standby applied it and stands behind.
                opened.says(batch_of(&tail[..1], head, now)).await;
                assert_eq!(opened.frame().await, ack(&tail[0]));
                assert_eq!(synced(), !stated, "a frame short of the head");
                // The frame that ends there leaves it at the head.
                opened.says(batch_of(&tail[1..], head, now)).await;
                assert_eq!(opened.frame().await, ack(&tail[2]));
                assert!(synced(), "the frame that reached the head");
                opened.refuses(Status::aborted("deposed")).await;
            };
            let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
            assert_eq!((ended.next, ended.applied), (Next::Lease, true));
        }
    }

    #[tokio::test]
    async fn a_restored_standby_reads_synced_once_it_applied_what_was_written_during_the_send() {
        for tail in [false, true] {
            let mut bed = Bed::new().await;
            let (_elector, standing, standby) = standing_by(&bed, FIVE);
            let leader = led();
            let now = Instant::now();
            let parts = snapshot(&leader, now);
            let place = leader.1.head(Kinds::All);
            // A row written while the snapshot was sent: the last part
            // states a head one past the snapshot's place.
            leader.0.upsert(facts("http://b:8000", 1));
            let written = entries(&leader.1, place);
            let head = leader.1.head(Kinds::All).seq;
            standing.send_replace(bed.holds("leader-1", 1));
            let named = standby.shared.leader(None);
            let applied = AtomicBool::new(false);
            let script = async {
                let mut opened = bed.opened().await;
                assert!(matches!(opened.frame().await, Some(Frame::Hello(_))));
                opened.says(resync()).await;
                for part in &parts {
                    let mut part = part.clone();
                    part.head_seq = head;
                    opened.says(part).await;
                }
                if tail {
                    opened.says(batch_of(&written, head, now)).await;
                    let ack = Frame::Ack(proto::ReplicationAck { seq: head });
                    assert_eq!(opened.frame().await, Some(ack));
                }
                opened.refuses(Status::aborted("deposed")).await;
            };
            let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
            assert_eq!((ended.next, ended.applied), (Next::Lease, true));
            // Restored and behind what was written since, it is not synced:
            // the frame that brings that tail leaves it at the head.
            let synced = standby.shared.lag.age(Instant::now()).is_some();
            assert_eq!(synced, tail, "the tail applied: {tail}");
        }
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
        let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
        assert_eq!((ended.next, ended.applied), (Next::Lease, true));
        assert_eq!(standby.follower.place(), leader.1.head(Kinds::All));
        assert_eq!(start(&standby.shared.feed.log), start(&leader.1));
    }

    #[tokio::test]
    async fn a_resync_is_the_leaders_snapshot_in_parts_on_the_stream_it_holds() {
        let mut bed = Bed::new().await;
        let (_elector, standing, standby) = standing_by(&bed, FIVE);
        let leader = led();
        let now = Instant::now();
        let b = leader.0.upsert(facts("http://b:8000", 1)).id();
        leader.0.record(b, 0, running(1), now, LoadSource::Stream);
        let parts = snapshot(&leader, now);
        assert_eq!(parts.len(), 2, "the rows' part and the load rows'");
        let place = leader.1.head(Kinds::All);
        standing.send_replace(bed.holds("leader-1", 1));
        let named = standby.shared.leader(None);
        let applied = AtomicBool::new(false);
        let script = async {
            let mut opened = bed.opened().await;
            assert!(matches!(opened.frame().await, Some(Frame::Hello(_))));
            // The leader's word, then its snapshot, a part a batch.
            opened.says(resync()).await;
            for part in &parts {
                opened.says(part.clone()).await;
            }
            // The log from the snapshot's place, on the same stream: the
            // standby applies it there and acknowledges it.
            leader.0.upsert(facts("http://c:8000", 1));
            for entry in entries(&leader.1, place) {
                opened.says(batch(&entry, now)).await;
                let seq = entry.at.seq;
                let ack = Frame::Ack(proto::ReplicationAck { seq });
                assert_eq!(opened.frame().await, Some(ack));
            }
            opened.refuses(Status::aborted("deposed")).await;
        };
        let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
        assert_eq!((ended.next, ended.applied), (Next::Lease, true));
        assert_eq!(standby.follower.place(), leader.1.head(Kinds::All));
        assert_eq!(held(&standby.shared.store), held(&leader.0));
        let lag = standby.shared.lag.age(Instant::now());
        assert!(lag.is_some(), "a restore is a frame applied");
        // One stream and no other call: the snapshot rode it. The leader's
        // word is no ask of the standby's.
        assert_eq!(*bed.names.lock(), ["standby-0"]);
        assert_eq!(standby.shared.lag.asks(), 0);
    }

    #[tokio::test]
    async fn a_stream_that_ends_inside_a_snapshot_leaves_the_store_as_it_was() {
        let mut bed = Bed::new().await;
        let (_elector, standing, standby) = standing_by(&bed, FIVE);
        // The standby follows a leader's first writes.
        let leader = led();
        let now = Instant::now();
        follow(&standby.follower, &leader.1, now);
        let place = standby.follower.place();
        let before = held(&standby.shared.store);
        // The leader moves on, and its snapshot is two parts.
        let b = leader.0.upsert(facts("http://b:8000", 1)).id();
        leader.0.record(b, 0, running(1), now, LoadSource::Stream);
        let parts = snapshot(&leader, now);
        assert_eq!(parts.len(), 2);
        let elsewhere = proto::ReplicationBatch {
            first_seq: parts[1].first_seq + 1,
            ..parts[1].clone()
        };
        let event = entries(&leader.1, place).remove(0);
        // After the first part: the stream's end, the same part again, a
        // part at another place, a batch of events.
        let cut_short = [
            ("the stream ends", Err(Status::unavailable("gone"))),
            ("a part out of its order", Ok(parts[0].clone())),
            ("a part at another place", Ok(elsewhere)),
            ("a batch of events", Ok(batch(&event, now))),
        ];
        standing.send_replace(bed.holds("leader-1", 1));
        let named = standby.shared.leader(None);
        for (what, next) in cut_short {
            let applied = AtomicBool::new(false);
            let script = async {
                let opened = bed.opened().await;
                opened.says(resync()).await;
                opened.says(parts[0].clone()).await;
                let sent = opened.batches.send(next).await;
                assert!(sent.is_ok(), "the stream stands");
                opened
            };
            let (ended, _opened) = tokio::join!(standby.stream(&named, &applied), script);
            assert_eq!((ended.next, ended.applied), (Next::Wait, false), "{what}");
            assert_eq!(standby.follower.place(), place, "{what}");
            assert_eq!(held(&standby.shared.store), before, "{what}");
        }
        // The whole snapshot on the next stream restores it. The word said
        // again starts the staging over: the parts after it are a
        // snapshot's from its first.
        let applied = AtomicBool::new(false);
        let script = async {
            let opened = bed.opened().await;
            opened.says(resync()).await;
            opened.says(parts[0].clone()).await;
            opened.says(resync()).await;
            for part in &parts {
                opened.says(part.clone()).await;
            }
            opened.refuses(Status::aborted("deposed")).await;
        };
        let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
        assert_eq!((ended.next, ended.applied), (Next::Lease, true));
        assert_eq!(held(&standby.shared.store), held(&leader.0));
    }

    #[tokio::test]
    async fn an_event_that_does_not_land_asks_a_restore_and_two_in_a_row_wait_a_streak_apart() {
        let mut bed = Bed::new().await;
        let (_elector, standing, standby) = standing_by(&bed, FIVE);
        let leader = led();
        let now = Instant::now();
        follow(&standby.follower, &leader.1, now);
        let place = standby.follower.place();
        // The leader's event after the next: past a gap, it does not land.
        let gap = |leader: &Replica, after: Position| {
            leader
                .0
                .upsert(facts(&format!("http://g{}:8000", after.seq), 1));
            leader
                .0
                .upsert(facts(&format!("http://h{}:8000", after.seq), 1));
            let past = entries(&leader.1, after).pop().expect("the leader wrote");
            batch(&past, now)
        };
        standing.send_replace(bed.holds("leader-1", 7));
        let named = standby.shared.leader(None);
        let hello = |seq, term| {
            Some(Frame::Hello(proto::ReplicaHello {
                replica_id: "standby-0".to_owned(),
                last_applied_seq: seq,
                term_seen: 7,
                max_backlog: 0,
                last_applied_term: term,
            }))
        };

        // The first: the stream is left, the ask is counted, and the next
        // stream opens at once.
        let lost = gap(&leader, place);
        let applied = AtomicBool::new(false);
        let script = async {
            let mut opened = bed.opened().await;
            assert_eq!(opened.frame().await, hello(place.seq, place.term));
            opened.says(lost).await;
            opened
        };
        let (ended, _opened) = tokio::join!(standby.stream(&named, &applied), script);
        assert_eq!((ended.next, ended.applied), (Next::Join, false));
        assert!(ended.why.is_none());
        assert_eq!(standby.shared.lag.asks(), 1);

        // The next hello asks: its head's number under no term, the term
        // it saw as it is. The leader answers its snapshot, and the event
        // after it again does not land: a streak's wait, not a stream.
        let parts = snapshot(&leader, now);
        let restored = leader.1.head(Kinds::All);
        let lost = gap(&leader, restored);
        let applied = AtomicBool::new(false);
        let script = async {
            let mut opened = bed.opened().await;
            assert_eq!(opened.frame().await, hello(place.seq, 0));
            opened.says(resync()).await;
            for part in &parts {
                opened.says(part.clone()).await;
            }
            opened.says(lost).await;
            opened
        };
        let (ended, _opened) = tokio::join!(standby.stream(&named, &applied), script);
        assert_eq!((ended.next, ended.applied), (Next::Wait, false));
        let why = "restored twice with no event landed between";
        assert_eq!(ended.why.as_deref(), Some(why));
        assert_eq!(standby.follower.place(), restored);
        assert_eq!(standby.shared.lag.asks(), 2);
        let first = standby.pace.due.lock().expect("a restore was asked");

        // It asks again, a beat after its last ask at the soonest.
        let applied = AtomicBool::new(false);
        let script = async {
            let mut opened = bed.opened().await;
            assert_eq!(opened.frame().await, hello(restored.seq, 0));
            opened.refuses(Status::aborted("deposed")).await;
        };
        let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
        assert_eq!((ended.next, ended.applied), (Next::Lease, false));
        let second = standby.pace.due.lock().expect("a restore was asked");
        assert!(second >= first + RESYNC, "{first:?} then {second:?}");
        // A restore, an event that lands, then one that does not: the pair
        // is over, and the next stream opens at once.
        let parts = snapshot(&leader, now);
        let restored = leader.1.head(Kinds::All);
        leader.0.upsert(facts("http://landed:8000", 1));
        let lands = entries(&leader.1, restored).remove(0);
        let lost = gap(&leader, lands.at);
        let applied = AtomicBool::new(false);
        let script = async {
            let mut opened = bed.opened().await;
            assert!(matches!(opened.frame().await, Some(Frame::Hello(_))));
            opened.says(resync()).await;
            for part in &parts {
                opened.says(part.clone()).await;
            }
            opened.says(batch(&lands, now)).await;
            opened.says(lost).await;
            opened
        };
        let (ended, _opened) = tokio::join!(standby.stream(&named, &applied), script);
        assert_eq!((ended.next, ended.applied), (Next::Join, true));
        assert_eq!(standby.follower.place(), lands.at);
        assert_eq!(standby.shared.lag.asks(), 3);
        assert_eq!(*bed.names.lock(), ["standby-0"; 4]);
    }

    #[tokio::test]
    async fn an_ask_states_the_floor_its_place_no_longer_names() {
        let mut bed = Bed::new().await;
        let (_elector, standing, standby) = standing_by(&bed, FIVE);
        // The standby followed a leader into term 9, and the lease it reads
        // names a leader of term 7.
        let leader = led();
        leader.0.set_term(9);
        follow(&standby.follower, &leader.1, Instant::now());
        let place = standby.follower.place();
        assert_eq!(place.term, 9);
        standing.send_replace(bed.holds("leader-1", 7));
        let named = standby.shared.leader(None);
        standby.asks.store(true, Ordering::Relaxed);
        let applied = AtomicBool::new(false);
        let script = async {
            let mut opened = bed.opened().await;
            let hello = proto::ReplicaHello {
                replica_id: "standby-0".to_owned(),
                last_applied_seq: place.seq,
                term_seen: 9,
                max_backlog: 0,
                last_applied_term: 0,
            };
            assert_eq!(opened.frame().await, Some(Frame::Hello(hello)));
            opened.refuses(Status::failed_precondition("deposed")).await;
        };
        let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
        assert_eq!((ended.next, ended.applied), (Next::Lease, false));
    }

    #[tokio::test]
    async fn a_restore_ends_the_ask_and_the_next_hello_names_the_place_restored() {
        let mut bed = Bed::new().await;
        let (_elector, standing, standby) = standing_by(&bed, FIVE);
        let leader = led();
        let now = Instant::now();
        let parts = snapshot(&leader, now);
        let restored = leader.1.head(Kinds::All);
        standing.send_replace(bed.holds("leader-1", 1));
        let named = standby.shared.leader(None);
        let hello = |place: Position| {
            Some(Frame::Hello(proto::ReplicaHello {
                replica_id: "standby-0".to_owned(),
                last_applied_seq: place.seq,
                term_seen: 1,
                max_backlog: 0,
                last_applied_term: place.term,
            }))
        };
        // An event did not land: the hello asks, and the snapshot answers.
        standby.asks.store(true, Ordering::Relaxed);
        let asking = standby.follower.asking();
        assert!(asking.seq > 0 && asking.term == 0, "{asking:?}");
        let applied = AtomicBool::new(false);
        let script = async {
            let mut opened = bed.opened().await;
            assert_eq!(opened.frame().await, hello(asking));
            opened.says(resync()).await;
            for part in &parts {
                opened.says(part.clone()).await;
            }
            opened.refuses(Status::unavailable("gone")).await;
        };
        let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
        assert_eq!((ended.next, ended.applied), (Next::Wait, true));
        // Restored, it names its place under its term again.
        let applied = AtomicBool::new(false);
        let script = async {
            let mut opened = bed.opened().await;
            assert_eq!(opened.frame().await, hello(restored));
            opened.refuses(Status::aborted("deposed")).await;
        };
        let (ended, ()) = tokio::join!(standby.stream(&named, &applied), script);
        assert_eq!((ended.next, ended.applied), (Next::Lease, false));
    }

    #[tokio::test]
    async fn a_snapshot_the_log_does_not_follow_is_left_and_the_lease_read_again() {
        let mut bed = Bed::new().await;
        let (_elector, standing, standby) = standing_by(&bed, FIVE);
        // The standby followed a leader into term 9.
        let leader = led();
        leader.0.set_term(9);
        let now = Instant::now();
        follow(&standby.follower, &leader.1, now);
        let place = standby.follower.place();
        let before = held(&standby.shared.store);
        // A replica that still believes it leads term 1 restores it.
        let parts = snapshot(&led(), now);
        standing.send_replace(bed.holds("leader-1", 9));
        let named = standby.shared.leader(None);
        let applied = AtomicBool::new(false);
        let script = async {
            let opened = bed.opened().await;
            opened.says(resync()).await;
            for part in &parts {
                opened.says(part.clone()).await;
            }
            opened
        };
        let (ended, _opened) = tokio::join!(standby.stream(&named, &applied), script);
        assert_eq!((ended.next, ended.applied), (Next::Lease, false));
        let why = ended.why.expect("a reason");
        assert!(
            why.ends_with("which this replica's log does not follow"),
            "{why}"
        );
        assert_eq!(standby.follower.place(), place);
        assert_eq!(held(&standby.shared.store), before);
    }

    #[tokio::test]
    async fn a_replica_that_leads_by_now_asks_no_restore() {
        let mut bed = Bed::new().await;
        let (mut elector, standing, standby) = standing_by(&bed, FIVE);
        standing.send_replace(bed.holds("leader-1", 1));
        let named = standby.shared.leader(None);
        // An event did not land, and the replica took its own lease since.
        standby.asks.store(true, Ordering::Relaxed);
        let took = elector.tick(Instant::now());
        assert!(matches!(took, Ok(Tick::Acquired { .. })), "{took:?}");
        let applied = AtomicBool::new(false);
        let ended = standby.stream(&named, &applied).await;
        assert_eq!((ended.next, ended.applied), (Next::Lease, false));
        assert!(ended.why.is_none());
        assert!(bed.streams.try_recv().is_err(), "no stream was opened");
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
    async fn a_replica_that_led_is_not_synced_before_a_frame_of_its_present_leader() {
        let mut bed = Bed::new().await;
        let (mut elector, standing, standby) = standing_by(&bed, FIVE);
        let shared = Arc::clone(&standby.shared);
        let held = || shared.lag.age(Instant::now()).is_some();
        let hello = |frame: Option<Frame>| matches!(frame, Some(Frame::Hello(_)));
        let (stop, stopped) = watch::channel(false);
        let leader = led();
        standing.send_replace(bed.holds("leader-1", 1));
        let mut running = JoinSet::new();
        running.spawn(standby.run(stopped));
        // The leader's events, acknowledged: the lag holds a frame.
        let mut first = bed.opened().await;
        assert!(hello(first.frame().await));
        let mut place = start(&leader.1);
        for entry in entries(&leader.1, place) {
            first.says(batch(&entry, Instant::now())).await;
            place = entry.at;
            let ack = Frame::Ack(proto::ReplicationAck { seq: place.seq });
            assert_eq!(first.frame().await, Some(ack));
        }
        assert!(held());

        // The lease names another leader: the replica held no lease
        // between the two, and its frame stands.
        standing.send_replace(bed.holds("leader-2", 2));
        let mut second = bed.opened().await;
        assert!(hello(second.frame().await));
        assert_eq!(first.frame().await, None, "the first stream is left");
        assert!(held(), "a follower's frame outlives a change of leader");

        // This replica takes the lease: the frame it applied before its
        // lead is forgotten.
        let took = elector.tick(Instant::now());
        assert!(matches!(took, Ok(Tick::Acquired { .. })), "{took:?}");
        standing.send_replace(Standing {
            holder: OWN.to_owned(),
            term: 3,
            looked: true,
        });
        assert_eq!(second.frame().await, None, "a leader follows nobody");
        assert!(!held(), "a replica that led holds no frame");

        // It leaves the lease and follows the next holder: not synced
        // before a frame of that leader, synced at it.
        assert!(elector.release().expect("the release"));
        standing.send_replace(bed.holds("leader-3", 4));
        let mut third = bed.opened().await;
        assert!(hello(third.frame().await));
        assert!(!held(), "no frame of its present leader yet");
        leader.0.upsert(facts("http://b:8000", 1));
        for entry in entries(&leader.1, place) {
            third.says(batch(&entry, Instant::now())).await;
            let ack = Frame::Ack(proto::ReplicationAck { seq: entry.at.seq });
            assert_eq!(third.frame().await, Some(ack));
        }
        assert!(held());
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
        let left = async { tokio::join!(standby.stream(&named, &applied), script) };
        let (ended, opened) = timeout(FIVE, left).await.expect("the stream is left");
        assert_eq!((ended.next, ended.applied), (Next::Wait, true));
        let why = "the leader sent no frame for 100ms";
        assert_eq!(ended.why.as_deref(), Some(why));
        drop(opened);
        // A snapshot whose next part does not arrive: the same bound.
        let applied = AtomicBool::new(false);
        let script = async {
            let opened = bed.opened().await;
            opened.says(resync()).await;
            opened
        };
        let left = async { tokio::join!(standby.stream(&named, &applied), script) };
        let (ended, held) = timeout(FIVE, left).await.expect("the stream is left");
        assert_eq!((ended.next, ended.applied), (Next::Wait, false));
        let why = "the leader sent no frame for 100ms";
        assert_eq!(ended.why.as_deref(), Some(why));
        drop(held);
        // A leader that opens no stream.
        bed.mute.store(true, Ordering::Relaxed);
        let applied = AtomicBool::new(false);
        let left = standby.stream(&named, &applied);
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
