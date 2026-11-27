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
//! replicated tables while a replica follows.
//!
//! A membership event and a restore go through the holder of the replica's
//! worker objects, which lays each row's object and hands the store the
//! write: a standby's objects stand at the place its log stands at too.

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
    election::ElectionObservation,
    generations::Generations,
    principal::ENTRY,
    replication::BEAT,
    service::{Leader, Shared},
    worker_wire::{self, Refused},
};
use crate::{
    engine::view::fingerprint,
    service::{
        proto::{
            self, grand_central_client::GrandCentralClient, replica_frame::Frame,
            replication_event::Event as Sent,
        },
        replication::{Log, LogPositionLost},
    },
    store::{LeasesStore, MemoryStore, Position, StoreCheckpoint, WorkerRow},
    worker_host::{Members, Objects},
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

/// An event's refusal where this build, or an applier with no door for
/// it, does not apply its kind.
const UNAPPLIED: Refused = Refused {
    path: "batch.events",
    why: "an event this build does not apply",
};

/// The standby's applier over one store, the log it journals to and the
/// holder of the replica's worker objects.
pub(crate) struct Follower {
    store: Arc<MemoryStore>,
    log: Arc<Log>,
    /// What lays a told row's object and hands the store the write.
    objects: Arc<Objects<Members>>,
    /// The door a leader's generation is applied through
    /// ([`Follower::through`]).
    generations: Option<Arc<Generations>>,
}

impl Follower {
    pub(crate) fn new(
        store: Arc<MemoryStore>,
        log: Arc<Log>,
        objects: Arc<Objects<Members>>,
    ) -> Self {
        Self {
            store,
            log,
            objects,
            generations: None,
        }
    }

    /// The same, applying a leader's generations through `generations`, the
    /// replica's one door for a document: the log is that door's one writer
    /// while the replica follows. With no door a generation's event is one
    /// this applier does not apply, and the standby leaves the synced state
    /// at its place.
    #[must_use]
    pub(crate) fn through(mut self, generations: Arc<Generations>) -> Self {
        self.generations = Some(generations);
        self
    }

    /// Where the standby stands: the place its next hello names. A standby
    /// that applied nothing names no place of any run, zero for both: its
    /// log stands at the place its own run drew, and its leader continues
    /// it from the leader's first write.
    pub(crate) fn place(&self) -> Position {
        let head = self.log.head();
        if self.log.first() == Some(head) {
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
            seq: self.log.head().seq.max(1),
            term: 0,
        }
    }

    /// The store restored from the leader's snapshot, its parts staged
    /// whole ([`Staged`]): the standby stands at the cut's place after it.
    /// `false` where the log refuses that place (a history that is not
    /// ahead of its own), the store as it was.
    ///
    /// The restore runs in the turn of the holder of the worker objects
    /// ([`Objects::restore_from`]): the cut's rows have their objects when
    /// it returns.
    ///
    /// `told` is the leader's standing generation where the snapshot
    /// states one. Its document's pass comes first, under the door's lock,
    /// and the generation lands behind the restore under that lock still:
    /// a document this replica does not run leaves the store as it was, by
    /// its path. A snapshot stands at one place for all it holds, so the
    /// generation journals none. With no door the generation is not
    /// applied.
    pub(crate) fn restore(
        &self,
        cut: StoreCheckpoint,
        told: Option<&proto::GenerationEvent>,
    ) -> Result<bool, Refused> {
        let following = match (told, &self.generations) {
            (Some(told), Some(generations)) => Some(generations.following(told)?),
            _ => None,
        };
        if !self.objects.restore_from(&self.store, cut) {
            return Ok(false);
        }
        following.map_or(Ok(()), |following| following.land(None))?;
        Ok(true)
    }

    /// One batch applied at `now`: each event at its position under the
    /// batch's term, by the store's door for a followed write of its kind,
    /// which alone lands there: a write of the store's own that comes
    /// between takes no part in it, and the event does not land. A load row
    /// is an older leader's: the store keeps none, and its place alone is
    /// journaled. The first event that does not
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
                Err(LogPositionLost) if matches!(event.event, Some(Sent::TermStart(_))) => {
                    self.log
                        .join(at)
                        .map_err(|LogPositionLost| Halt::Lost(at))?;
                    self.store.follow_term(at);
                    continue;
                }
                Err(LogPositionLost) => return Err(Halt::Lost(at)),
            };
            match &event.event {
                Some(Sent::Membership(told)) => {
                    let (epoch, change, row) =
                        worker_wire::told_change(told, now).map_err(Halt::Unread)?;
                    // Through the holder: the row's object is built, the
                    // store takes the write, and the object is put.
                    self.objects.follow_at(&self.store, at, epoch, change, row);
                }
                // A leader of this build sends no load row: this one is an
                // older leader's. The store keeps no load row, so the place
                // alone is journaled and the next event follows.
                Some(Sent::Load(_)) => adopting.mark(),
                Some(Sent::TermStart(_)) => self.store.follow_term(at),
                // Through the replica's door, which journals the place: the
                // document's pass runs here, on the applier's own task.
                Some(Sent::Generation(told)) => {
                    let Some(generations) = &self.generations else {
                        return Err(Halt::Unread(UNAPPLIED));
                    };
                    generations
                        .following(told)
                        .and_then(|following| following.land(Some(at)))
                        .map_err(Halt::Unread)?;
                }
                // A kind this build does not apply, and one it does not know.
                Some(_) | None => return Err(Halt::Unread(UNAPPLIED)),
            }
            if !adopting.landed() {
                return Err(Halt::Lost(at));
            }
        }
        Ok(())
    }
}

/// A leader's snapshot as its parts arrive: the rows, held apart from the
/// store until the last part. A part's load rows are an older leader's and
/// are read past. The store is restored
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
    /// The leader's standing generation, as the first part that states one
    /// said it: the last part hands it on with the cut.
    generation: Option<proto::GenerationEvent>,
}

impl Staged {
    /// One batch of the stream inside a snapshot, read at `now`: the
    /// snapshot's next part, staged, and at the last part the snapshot
    /// whole, as [`Follower::restore`] takes it: the cut the store is
    /// restored from, and the leader's standing generation where a part
    /// stated one. A part is its snapshot's next where its
    /// batch carries no event and says the place and the epoch the first
    /// part said, under the next number. Anything else is refused by its
    /// path, and the stream is left with nothing restored.
    pub(crate) fn take(
        &mut self,
        batch: &proto::ReplicationBatch,
        now: Instant,
    ) -> Result<Option<(StoreCheckpoint, Option<proto::GenerationEvent>)>, Refused> {
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
        if self.generation.is_none() {
            self.generation = part.generation.as_deref().cloned();
        }
        worker_wire::told_part(part, now, &mut self.rows)?;
        Ok(part.last.then(|| {
            let cut = StoreCheckpoint {
                at,
                epoch: part.membership_epoch,
                rows: std::mem::take(&mut self.rows),
            };
            (cut, self.generation.take())
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
    /// `0` before any, and once this replica has led since or its lease
    /// has named another leader's term.
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
    /// one. The standby loop's call once this replica has held the lease,
    /// and once its lease names another leader's term: a frame it applied
    /// before says nothing of the leader it follows next.
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

/// The lag a standby reads synced under: one load poll interval, and never
/// under two of the leader's heartbeats, which an idle stream's lag reaches
/// by itself.
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
async fn pause(steps: &mut watch::Receiver<ElectionObservation>, lease: bool, wait: Duration) {
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
        let follower = Follower::new(
            Arc::clone(&shared.store),
            Arc::clone(&shared.feed.log),
            Arc::clone(&shared.objects),
        )
        .through(Arc::clone(&shared.generations));
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
    /// frame of the leader it follows next, and so is one whose lease
    /// names another leader, or the same one under another term: the sync
    /// is with one leader's term ([`Lag::forget`]).
    pub(crate) async fn run(self, mut stop: watch::Receiver<bool>) {
        let mut steps = self.shared.standing.clone();
        let mut streak = Streak::default();
        let mut led = self.shared.fence.term();
        // The term of the leader the last stream was opened against.
        let mut followed = None;
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
                // A term names one leader. A frame applied under another
                // says nothing of the one the lease names now: this
                // replica reads not synced until that leader's first.
                if followed.replace(leader.term) != Some(leader.term) {
                    self.shared.lag.forget();
                }
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
    async fn turned(
        &self,
        steps: &mut watch::Receiver<ElectionObservation>,
        leader: &Leader,
    ) -> Next {
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
            Ok(channel) => GrandCentralClient::new(channel),
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
                let (cut, generation) = match staging.take(&batch, now) {
                    Ok(Some(whole)) => whole,
                    Ok(None) => continue,
                    Err(refused) => return Ended::waits(landed, unread(refused)),
                };
                staged = None;
                let (at, rows) = (cut.at, cut.rows.len());
                match self.follower.restore(cut, generation.as_ref()) {
                    Ok(true) => {}
                    Ok(false) => return Ended::unfollowed(at, landed),
                    Err(refused) => return Ended::waits(landed, unread(refused)),
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
                    Err(Halt::Lost(at)) => {
                        // The events of the batch ahead of that one landed
                        // and stand, as a whole batch's would: an event
                        // followed the last restore.
                        let partly = at.seq > batch.first_seq;
                        if partly {
                            self.fresh.store(false, Ordering::Relaxed);
                        }
                        return self.lost(leader, at, landed || partly);
                    }
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
            // every applied frame did. A place is the leader's under a
            // term alone: a row of the standby's own stands under none,
            // and a head stated at its number is another event's.
            let place = self.follower.place();
            if batch.head_seq == 0 || (place.term != 0 && batch.head_seq == place.seq) {
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
    /// speed. The sync is forgotten here: a standby an event did not land
    /// on holds less than its leader does, whatever its last frame said.
    /// The restore's last part sets it again, or the frame that brings the
    /// restore's tail.
    fn lost(&self, leader: &Leader, at: Position, landed: bool) -> Ended {
        self.asks.store(true, Ordering::Relaxed);
        self.shared.lag.asked();
        self.shared.lag.forget();
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
#[path = "../../../tests/unit/service/controller/standby/tests.rs"]
mod tests;
