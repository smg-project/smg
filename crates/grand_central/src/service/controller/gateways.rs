//! Who is present (design 4.3, the stream rule): a gateway is present from
//! its `Welcome` until its grace period ends with no new `Hello`, one
//! period more at most while a session a newer one replaced still runs,
//! and what the store holds under its key lives exactly that long: its
//! lease records and its vetoes.
//!
//! Presence is read from the clock at the instant of the question, as the
//! fence is: a gateway is present while a session of it stands and for the
//! grace period after the last one ended, or one period more at most while
//! a session it replaced still runs. No task tells a reader who is present.
//! The sweep only lets go what a gateway that ended held, so a late sweep
//! changes no answer.
//!
//! Three rules come from the fence (design 5.3). A replica whose fence is
//! closed lets nothing go: a fenced leader releases nothing. At a take
//! every gone gateway's period starts again: no `Hello` could be welcomed
//! here while this replica did not lead, so that time is not the gateway's
//! silence. And a `Hello` is seated under the term the table's own read of
//! the fence shows, under its lock, or not at all: one read under a term
//! the election has left supersedes no session of the term that stands.
//!
//! A `Welcome` and the letting-go of one gateway are ordered by the table's
//! lock, held across the store's release: a gateway is welcomed with what
//! it held, or let go and welcomed with nothing, never welcomed and then
//! let go. The table's lock is taken before any lock of the store.
//!
//! A session that was replaced applies no frame from the instant it is
//! told: the session asks its seat before each frame, under no lock. A
//! frame already past that question may still be written, so a gateway is
//! not let go while a session it replaced still runs: what that frame
//! writes lands under the row it was read for and leaves with it. A
//! replaced session ends at its next turn; one that never ends holds the
//! row for one period more and no longer. What it writes after that
//! lands under a key no row holds, and stands until the gateway is
//! welcomed and let go again.
//!
//! A gateway's vetoes are noted here by kind (design 5.1). The store shows
//! one veto a gateway and worker, with no kind and two ends. The first is
//! the latest of the kinds' ends that still stand, so a kind that clears
//! never lifts another. The second is the veto's hold: the latest end of
//! the kinds no rule lifts ([`hold_of`]). A session writes a
//! veto through its seat, under the table's lock, where the question is
//! asked again: a replaced session's veto is never written, and no pass
//! reclaims one. The note follows the store: a veto is about a worker's
//! row and leaves with it, and the presence's end lifts every veto with
//! the leases, in the store's one drop.
//!
//! The table holds two bounds of its own. A ceiling of rows: a `Hello` that
//! would make a new row past it is refused, so the names a listener lets in
//! never grow the table without end; a present gateway's `Hello` makes no
//! row and is always taken. And one subject a key: the store keys a gateway
//! by a fingerprint of its subject, so a `Hello` of another subject under a
//! present gateway's key is refused, never merged into that gateway's row.

use std::{
    collections::BTreeSet,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Weak,
    },
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use tokio::{
    sync::{watch, Notify},
    time::MissedTickBehavior,
};
use tracing::{info, warn};

use crate::{
    engine::view::WorkerId,
    service::{elector::Fence, proto::gateway_veto::Kind},
    store::{GatewayId, HealthSink, LeasesStore, MemoryStore, VetoWrite, WorkersStore},
};

/// A worker's ends, one a kind's number: most hold one or two.
type Ends = Vec<(i32, Instant)>;

/// The latest of a worker's ends: what the store shows.
fn latest(ends: &[(i32, Instant)]) -> Option<Instant> {
    ends.iter().map(|&(_, end)| end).max()
}

/// The kind the all-vetoed rule lifts as its last resort: the contract's
/// `KIND_LIVENESS`, a doubt of the worker's liveness.
const LIVENESS: i32 = Kind::Liveness as i32;

/// The hold among a worker's ends, which the store shows beside the latest:
/// the latest end of the kinds no rule lifts. Every kind but [`LIVENESS`]
/// holds. A breaker's open window does, and so does a number this build
/// has no name for: the rule lifts what it can name.
fn hold_of(ends: &[(i32, Instant)]) -> Option<Instant> {
    let holding = ends.iter().filter(|&&(kind, _)| kind != LIVENESS);
    holding.map(|&(_, end)| end).max()
}

/// How many kinds of veto one worker holds for one gateway at most. A kind
/// is kept by its number, so without a ceiling the kinds a worker holds
/// would be a peer's to choose. The contract names two.
pub(crate) const KINDS_MAX: usize = 8;

/// How many vetoes one gateway holds at most, over all its workers: what a
/// `Hello` lists whole. A frame is held to it, so every state a session
/// reaches is one its next `Hello` can carry.
pub(crate) const VETOES_MAX: usize = 4_096;

/// Whether a list holds more than [`KINDS_MAX`] kinds for one worker: a
/// `Hello` that lists so is refused whole, before anything is applied.
pub(crate) fn over_kinds(list: impl Iterator<Item = (WorkerId, i32)>) -> bool {
    let mut kinds: FxHashMap<WorkerId, Vec<i32>> = FxHashMap::default();
    for (worker, kind) in list {
        let held = kinds.entry(worker).or_default();
        if !held.contains(&kind) {
            if held.len() >= KINDS_MAX {
                return true;
            }
            held.push(kind);
        }
    }
    false
}

/// What became of a veto a session brought.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Vetoed {
    /// The store shows the latest of the worker's ends that still stand,
    /// or no veto when none does.
    Shown,
    /// The store holds no row for the worker: the veto is dropped and
    /// nothing of it is kept.
    NoWorker,
    /// The session is not its gateway's standing one: nothing is written.
    Replaced,
    /// The worker holds [`KINDS_MAX`] kinds for the gateway and the veto
    /// would set one more: nothing is written, and what stands stays.
    Full,
    /// The gateway holds [`VETOES_MAX`] vetoes and the veto would set one
    /// more: nothing is written, and what stands stays.
    Over,
}

/// Why a `Hello` takes no seat. A refused `Hello` changes nothing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Refused {
    /// A gateway of another subject is present under the same key.
    AnotherSubject,
    /// The table has numbered every session it can.
    Exhausted,
    /// The table holds its ceiling of rows and this `Hello` would make a
    /// new one.
    Full { ceiling: usize },
    /// The table's read of the fence does not show the term the session
    /// read for this `Hello`: the fence is closed (`None`), or open under
    /// a later term.
    Stepped { read: u64, shown: Option<u64> },
}

/// The presence table.
pub(crate) struct Gateways {
    /// How long a gateway stays present after its last session ended.
    grace: Duration,
    /// How many rows the table holds at most: the gateways with a standing
    /// session or inside their grace period.
    ceiling: usize,
    fence: Arc<Fence>,
    store: Arc<MemoryStore>,
    table: Mutex<Table>,
}

#[derive(Default)]
struct Table {
    rows: FxHashMap<GatewayId, Row>,
    /// The next instant each row needs a sweep visit, ordered by instant.
    /// A row has at most one entry, kept in step with [`Row::deadline`].
    deadlines: BTreeSet<(Instant, GatewayId)>,
    /// A case reads the sweep's visits, and the sweep itself counts them here.
    #[cfg(test)]
    sweep_visits: usize,
    /// The highest term the table was read under since this replica last
    /// led; `None` when the last read found the fence closed, and before
    /// the first.
    led: Option<u64>,
    /// Numbers every session the table welcomed and never restarts: a
    /// session's number is its alone, whatever became of its gateway's row.
    sessions: u64,
}

struct Row {
    /// The verified subject, for the log.
    subject: String,
    /// The number of the gateway's latest session, `0` before the first: a
    /// stream's end is the row's only while the number is that stream's own.
    session: u64,
    /// When the last session ended; `None` while one stands.
    gone: Option<Instant>,
    /// Wakes the standing session when a newer one takes its place.
    superseded: Arc<Replaced>,
    /// The sessions a newer one replaced that have not ended yet. The
    /// gateway is not let go under one: a frame past its question of the
    /// seat may still be written.
    replaced: Vec<Weak<Replaced>>,
    /// The gateway's vetoes by worker: a note of what the store was told,
    /// kind by kind. An end past its instant is no veto.
    vetoes: FxHashMap<WorkerId, Ends>,
    /// How many ends the note holds, over every worker.
    held: usize,
    /// No end the note holds is earlier than this instant: a write of an
    /// end lowers it, an end that leaves does not raise it, and a walk for
    /// ends past their instants ([`Row::prune`]) sets it to the earliest
    /// end left. `None` only while the note holds no end.
    next_end: Option<Instant>,
    /// This row's entry in [`Table::deadlines`].
    deadline: Option<Instant>,
    /// The membership epoch at the note's last walk for rows that left
    /// ([`Row::left`]), `None` before the first: a row leaves by a
    /// membership write alone, so one walk an epoch finds every one.
    left_at: Option<u64>,
    /// Held for as long as this presence lasts: a [`Presence`] waits for
    /// its drop.
    ended: watch::Sender<()>,
}

/// What a session is told when a newer one of its gateway takes its place.
/// The flag is asked before every frame, under no lock; the wake ends a
/// stream that waits.
#[derive(Default)]
struct Replaced {
    told: AtomicBool,
    wake: Notify,
}

/// A standing session's place in the table. Its drop is the session's end.
pub(crate) struct SessionPlace {
    gateways: Arc<Gateways>,
    gateway: GatewayId,
    session: u64,
    superseded: Arc<Replaced>,
    /// A case arms this pause, and the place's own drop waits on it.
    #[cfg(test)]
    drop_pause: Option<Arc<DropPause>>,
}

#[cfg(test)]
struct DropPause {
    gone: std::sync::Barrier,
    swept: std::sync::Barrier,
}

impl Row {
    /// An end past its instant is no veto: the note keeps none, and the
    /// count is the note's. The note is walked only once the earliest end
    /// it may hold is past: until then no end left it by its instant.
    fn prune(&mut self, now: Instant) {
        if self.next_end.is_none_or(|next| next > now) {
            return;
        }
        let mut held = 0;
        let mut next_end = None;
        self.vetoes.retain(|_, ends| {
            // One pass over a worker's ends drops the ends past and keeps
            // the earliest end left: one more compare an end kept.
            ends.retain(|&(_, end)| {
                let stands = end > now;
                if stands && next_end.is_none_or(|next: Instant| end < next) {
                    next_end = Some(end);
                }
                stands
            });
            held += ends.len();
            !ends.is_empty()
        });
        self.held = held;
        self.next_end = next_end;
    }

    /// The ends of workers whose rows left since they were noted go: the
    /// store shows no veto of `gateway` on them. One question a noted
    /// worker, asked where a gateway's ceiling stands in the way of a frame
    /// and nowhere else, once a membership epoch.
    fn left(&mut self, store: &MemoryStore, gateway: GatewayId, now: Instant) {
        self.vetoes
            .retain(|worker, _| store.vetoed(gateway, *worker, now));
        self.held = self.vetoes.values().map(Vec::len).sum();
    }
}

impl SessionPlace {
    /// Resolves when a newer session of the gateway took this one's place.
    pub(crate) async fn superseded(&self) {
        self.superseded.wake.notified().await;
    }

    /// Whether a newer session of the gateway took this one's place: asked
    /// before a frame is applied, under no lock.
    pub(crate) fn replaced(&self) -> bool {
        self.superseded.told.load(Ordering::Acquire)
    }

    /// The grace period of the table this seat stands in: the session's
    /// `Welcome` tells it in whole milliseconds.
    pub(crate) fn grace(&self) -> Duration {
        self.gateways.grace()
    }

    /// One kind's veto on `worker` set until `until`, or lifted, at `now`:
    /// the other kinds' ends stand, and the store shows the latest of them
    /// with the hold among them ([`hold_of`]).
    pub(crate) fn veto(
        &self,
        worker: WorkerId,
        kind: i32,
        until: Option<Instant>,
        now: Instant,
    ) -> Vetoed {
        self.gateways
            .veto(self.gateway, self.session, worker, kind, until, now)
    }

    /// The gateway's standing vetoes as its `Hello` lists them, whole: a
    /// veto the list omits is lifted. Returns how many ends are kept.
    pub(crate) fn restore(&self, vetoes: &[(WorkerId, i32, Instant)], now: Instant) -> usize {
        self.gateways
            .restore(self.gateway, self.session, vetoes, now)
    }
}

impl Drop for SessionPlace {
    fn drop(&mut self) {
        // A no-op after `end`: the row's period started there.
        self.gateways
            .gone(self.gateway, self.session, &self.superseded, Instant::now());
        #[cfg(test)]
        if let Some(pause) = &self.drop_pause {
            pause.gone.wait();
            pause.swept.wait();
        }
    }
}

/// A gateway's presence as a decide stream holds it.
#[derive(Clone)]
pub(crate) struct Presence(watch::Receiver<()>);

impl Presence {
    /// Whether the presence ended: one load, where a poll of
    /// [`Presence::ended`] takes the watch's lock.
    pub(crate) fn is_ended(&self) -> bool {
        self.0.is_closed()
    }

    /// Resolves when the presence ended.
    pub(crate) async fn ended(&mut self) {
        // No value is ever sent: the row holds the sender and goes with
        // the presence.
        while self.0.changed().await.is_ok() {}
    }
}

impl Gateways {
    pub(crate) fn new(
        grace: Duration,
        ceiling: usize,
        fence: Arc<Fence>,
        store: Arc<MemoryStore>,
    ) -> Self {
        Self {
            grace,
            ceiling,
            fence,
            store,
            table: Mutex::new(Table::default()),
        }
    }

    /// The grace period, as a `Welcome` tells it.
    pub(crate) fn grace(&self) -> Duration {
        self.grace
    }

    /// The next instant a row needs pruning or may be over. A gone row is
    /// first asked at one grace period; if a replaced session still runs,
    /// it is asked once more at the hard two-period limit.
    fn deadline(&self, row: &Row, now: Instant) -> Option<Instant> {
        let presence = row.gone.and_then(|gone| {
            let first = gone.checked_add(self.grace)?;
            let runs = row
                .replaced
                .iter()
                .any(|session| session.strong_count() > 0);
            if first <= now && runs {
                gone.checked_add(self.grace.saturating_mul(2))
            } else {
                Some(first)
            }
        });
        presence.into_iter().chain(row.next_end).min()
    }

    /// Replace one row's scheduled instant. Removing the old entry before
    /// adding the new one keeps the index bounded to one entry per row.
    fn set_deadline(table: &mut Table, gateway: GatewayId, deadline: Option<Instant>) {
        let Some(row) = table.rows.get_mut(&gateway) else {
            return;
        };
        if row.deadline == deadline {
            return;
        }
        if let Some(old) = std::mem::replace(&mut row.deadline, deadline) {
            table.deadlines.remove(&(old, gateway));
        }
        if let Some(deadline) = deadline {
            table.deadlines.insert((deadline, gateway));
        }
    }

    /// A `Welcome` at `now` under `term`, the session's read of the fence,
    /// which the table's own read under its lock must show too: the
    /// gateway is present, and the session takes its seat. A session of
    /// the gateway that still stands is superseded. A gateway whose period
    /// had ended before this `Hello`, the sweep not yet passed, is let go
    /// first: the new presence starts with nothing of the earlier one.
    ///
    /// # Errors
    ///
    /// The table has numbered every session it can: it is asked first, and
    /// nothing is read or changed. The fence no longer shows `term`: no seat
    /// is taken, no session is superseded and nothing is let go. Or a
    /// gateway of another subject is present under the key, or the row
    /// would be a new one past the table's ceiling. A row whose period is
    /// over counts until the sweep's next pass, a tenth of the period at
    /// most.
    pub(crate) fn welcome(
        self: &Arc<Self>,
        gateway: GatewayId,
        subject: &str,
        term: u64,
        now: Instant,
    ) -> Result<SessionPlace, Refused> {
        let mut table = self.table.lock();
        let Some(session) = table.sessions.checked_add(1) else {
            return Err(Refused::Exhausted);
        };
        let shown = self.fence.leader_term(now);
        self.observe(&mut table, shown, now);
        // The seat is taken under the term this read shows or not at all:
        // a Hello read under a term the election has left would supersede
        // a session of the term that stands.
        if shown != Some(term) {
            return Err(Refused::Stepped { read: term, shown });
        }
        if table
            .rows
            .get(&gateway)
            .is_some_and(|row| self.over(row, now))
        {
            Self::set_deadline(&mut table, gateway, None);
            if let Some(row) = table.rows.remove(&gateway) {
                self.end(vec![(gateway, row)]);
            }
        }
        match table.rows.get(&gateway) {
            // One key, one subject: two names of one fingerprint are never
            // one gateway. Nothing of the present gateway is touched.
            Some(row) if row.subject != subject => {
                warn!(
                    present = %row.subject,
                    refused = %subject,
                    "a Hello's subject shares its key with a present gateway of another subject: refused"
                );
                return Err(Refused::AnotherSubject);
            }
            // A new row past the ceiling: the table's bound that is not time.
            // The table has no gauge yet: this line and the refusal are its
            // signal.
            None if table.rows.len() >= self.ceiling => {
                warn!(
                    refused = %subject,
                    ceiling = self.ceiling,
                    "a Hello would make a new row past the table's ceiling of present gateways: refused"
                );
                return Err(Refused::Full {
                    ceiling: self.ceiling,
                });
            }
            _ => {}
        }
        table.sessions = session;
        let row = table.rows.entry(gateway).or_insert_with(|| Row {
            subject: subject.to_owned(),
            session: 0,
            gone: None,
            superseded: Arc::new(Replaced::default()),
            replaced: Vec::new(),
            vetoes: FxHashMap::default(),
            held: 0,
            next_end: None,
            deadline: None,
            left_at: None,
            ended: watch::channel(()).0,
        });
        if row.session > 0 && row.gone.is_none() {
            row.superseded.told.store(true, Ordering::Release);
            row.superseded.wake.notify_one();
            // The replaced session runs until it reads the word: the row
            // waits for its end.
            row.replaced.retain(|session| session.strong_count() > 0);
            row.replaced.push(Arc::downgrade(&row.superseded));
        }
        row.session = session;
        row.gone = None;
        row.superseded = Arc::new(Replaced::default());
        let superseded = Arc::clone(&row.superseded);
        let deadline = self.deadline(row, now);
        Self::set_deadline(&mut table, gateway, deadline);
        Ok(SessionPlace {
            gateways: Arc::clone(self),
            gateway,
            session,
            superseded,
            #[cfg(test)]
            drop_pause: None,
        })
    }

    /// The gateway's presence at `now` for a decide stream: `None` when it
    /// is not present. No decision asks: a stream does where it is placed
    /// under a term, at its open or at a request's receipt.
    pub(crate) fn presence(&self, gateway: GatewayId, now: Instant) -> Option<Presence> {
        let mut table = self.table.lock();
        self.observe(&mut table, self.fence.leader_term(now), now);
        let row = table.rows.get(&gateway)?;
        (!self.over(row, now)).then(|| Presence(row.ended.subscribe()))
    }

    /// One pass of the sweep at `now`: every gateway whose period is over
    /// is let go, in one write of the store for them all. Returns how many
    /// were let go. A replica that does not lead at `now` lets nothing go.
    pub(crate) fn sweep(&self, now: Instant) -> usize {
        let mut table = self.table.lock();
        if !self.observe(&mut table, self.fence.leader_term(now), now) {
            return 0;
        }
        let mut gone = Vec::new();
        while table
            .deadlines
            .first()
            .is_some_and(|(deadline, _)| *deadline <= now)
        {
            let Some((deadline, gateway)) = table.deadlines.pop_first() else {
                break;
            };
            #[cfg(test)]
            {
                table.sweep_visits += 1;
            }
            let Some(row) = table.rows.get_mut(&gateway) else {
                continue;
            };
            row.deadline = None;
            row.prune(now);
            if self.over(row, now) {
                if let Some(row) = table.rows.remove(&gateway) {
                    gone.push((gateway, row));
                }
            } else {
                let next = self.deadline(row, now);
                debug_assert_ne!(next, Some(deadline));
                Self::set_deadline(&mut table, gateway, next);
            }
        }
        let let_go = gone.len();
        self.end(gone);
        let_go
    }

    /// The sweep's task: a pass every tenth of the grace period until
    /// `stop`, on no stream's path. The leases' due ends ride it too
    /// ([`LeasesStore::expire`]): a file of leases whose decider waits for
    /// a job issues nothing, so nothing else would end its due leases. So
    /// does the upkeep of the engine that decides under the store's
    /// standing document: that engine runs no task of its own.
    pub(crate) async fn run(&self, mut stop: watch::Receiver<bool>) {
        let mut tick = tokio::time::interval((self.grace / 10).max(Duration::from_millis(1)));
        tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = tick.tick() => {
                    let now = Instant::now();
                    self.sweep(now);
                    // The names that left ride this clock: one that stayed
                    // away past its time takes its session pins with it.
                    self.store.end_gone(now);
                    // So do the leases past their dispatch deadline in
                    // a file whose decider waits for a job: one at work
                    // ends its own file's at its issues.
                    drop(LeasesStore::expire(&*self.store, now));
                    // So does the upkeep of the standing document's engine,
                    // which that engine's own task runs in a gateway's
                    // process: a session key past its end leaves its
                    // table, and so does the record of sent prefixes of a
                    // model no worker is left in. A replica runs it
                    // whether it leads or not.
                    self.store.deciding(|world| world.applied().sweep(now));
                }
                _ = stop.wait_for(|stop| *stop) => break,
            }
        }
    }

    /// The row of `gateway` while `session` is its standing session:
    /// `None` for a session that was replaced, or whose stream ended.
    fn standing(table: &mut Table, gateway: GatewayId, session: u64) -> Option<&mut Row> {
        table
            .rows
            .get_mut(&gateway)
            .filter(|row| row.session == session && row.gone.is_none())
    }

    /// A veto frame of the session `session` of `gateway`, under the
    /// table's lock: one kind's end on `worker` set or lifted, then one
    /// write of what the store shows.
    fn veto(
        &self,
        gateway: GatewayId,
        session: u64,
        worker: WorkerId,
        kind: i32,
        until: Option<Instant>,
        now: Instant,
    ) -> Vetoed {
        let mut table = self.table.lock();
        let Some(row) = Self::standing(&mut table, gateway, session) else {
            return Vetoed::Replaced;
        };
        let mut ends = row.vetoes.remove(&worker).unwrap_or_default();
        row.held -= ends.len();
        ends.retain(|&(of, end)| of != kind && end > now);
        let until = until.filter(|until| *until > now);
        // A worker removed since the last write took its vetoes with its
        // row, and one registered again starts clean: where the note says
        // an end stands and the store shows none, the note's ends go. The
        // store asks it under the hold its write takes, so no row leaves
        // and comes back between the question and the write.
        let mut stands = !ends.is_empty();
        let vetoed = loop {
            if until.is_some() {
                let kinds = ends.len() >= KINDS_MAX;
                if kinds || row.held + ends.len() >= VETOES_MAX {
                    // A ceiling stands in the way only while the note's
                    // ends stand: asked with nothing written.
                    if stands && !self.store.vetoed(gateway, worker, now) {
                        ends.clear();
                        stands = false;
                        continue;
                    }
                    // One kind more than a worker holds: the ends that
                    // stand are kept as they are and the store is not
                    // written.
                    if kinds {
                        break Vetoed::Full;
                    }
                    // One veto more than a gateway holds, once the ends
                    // past their instants are out of the count, and then
                    // the ends of rows that left: the ceiling counts the
                    // vetoes the store shows.
                    row.prune(now);
                    if row.held + ends.len() >= VETOES_MAX {
                        // A row leaves by a membership write alone and
                        // each moves the epoch, read here before the walk:
                        // the store is asked once an epoch, and a frame
                        // refused again under it asks nothing. The prune
                        // above walks the note only past its earliest end.
                        let epoch = WorkersStore::epoch(&*self.store);
                        if row.left_at != Some(epoch) {
                            row.left(&self.store, gateway, now);
                            row.left_at = Some(epoch);
                        }
                    }
                    if row.held + ends.len() >= VETOES_MAX {
                        break Vetoed::Over;
                    }
                }
            }
            let shown = latest(&ends).into_iter().chain(until).max();
            // The frame's own end is a hold's unless its kind is the one a
            // rule lifts.
            let own = until.filter(|_| kind != LIVENESS);
            let hold = hold_of(&ends).into_iter().chain(own).max();
            match self
                .store
                .gateway_veto_noted(gateway, worker, shown, hold, stands, now)
            {
                VetoWrite::Written => {
                    ends.extend(until.map(|until| (kind, until)));
                    break Vetoed::Shown;
                }
                // The row left since the note was written: its ends went
                // with it, and the frame is written again by itself.
                VetoWrite::Left => {
                    ends.clear();
                    stands = false;
                }
                // The veto is dropped and nothing of it is kept.
                VetoWrite::NoRow => {
                    ends.clear();
                    break Vetoed::NoWorker;
                }
            }
        };
        if !ends.is_empty() {
            row.held += ends.len();
            // The worker's ends were out of the note while a walk may have
            // set its earliest end: each of them lowers it again.
            let first = ends.iter().map(|&(_, end)| end).min();
            row.next_end = first.into_iter().chain(row.next_end).min();
            row.vetoes.insert(worker, ends);
        }
        let deadline = self.deadline(row, now);
        Self::set_deadline(&mut table, gateway, deadline);
        vetoed
    }

    /// A `Hello`'s list for the session `session` of `gateway`, under the
    /// table's lock: the note becomes the list, a veto the list omits is
    /// lifted in the store and each listed worker's latest end is shown.
    /// An end at or before `now` is no veto; of two entries of one worker
    /// and kind the later end stands. Returns how many ends are kept: none
    /// of a worker the store holds no row for.
    fn restore(
        &self,
        gateway: GatewayId,
        session: u64,
        vetoes: &[(WorkerId, i32, Instant)],
        now: Instant,
    ) -> usize {
        let mut table = self.table.lock();
        let Some(row) = Self::standing(&mut table, gateway, session) else {
            return 0;
        };
        let mut listed: FxHashMap<WorkerId, Ends> = FxHashMap::default();
        for &(worker, kind, end) in vetoes {
            if end <= now {
                continue;
            }
            let ends = listed.entry(worker).or_default();
            match ends.iter_mut().find(|(of, _)| *of == kind) {
                Some((_, stands)) => *stands = (*stands).max(end),
                None => ends.push((kind, end)),
            }
        }
        for worker in row.vetoes.keys() {
            if !listed.contains_key(worker) {
                self.store.gateway_veto(gateway, *worker, None);
            }
        }
        // The list is the gateway's word of this instant: no note stands
        // behind it, and a worker with no row keeps nothing.
        listed.retain(|worker, ends| {
            let written = self.store.gateway_veto_noted(
                gateway,
                *worker,
                latest(ends),
                hold_of(ends),
                false,
                now,
            );
            written == VetoWrite::Written
        });
        row.vetoes = listed;
        // The list replaced the note: the epoch kept for the old note's
        // last walk says nothing of this one, and the next frame at the
        // ceiling asks the store whatever the epoch reads by then.
        row.left_at = None;
        row.held = row.vetoes.values().map(Vec::len).sum();
        row.next_end = row.vetoes.values().flatten().map(|&(_, end)| end).min();
        let held = row.held;
        let deadline = self.deadline(row, now);
        Self::set_deadline(&mut table, gateway, deadline);
        held
    }

    /// Whether `term`, a read of the fence at `now`, shows a leader, with
    /// the take's rule applied to the table: at the first read under a term
    /// above the one the table was last read under, or after a read that
    /// found the fence closed, every gone gateway's period starts again at
    /// `now`.
    fn observe(&self, table: &mut Table, term: Option<u64>, now: Instant) -> bool {
        match (term, table.led) {
            // The term the table was last read under, or a read that lost
            // its race to a later one: nothing moved.
            (Some(term), Some(led)) if term <= led => {}
            (Some(_), _) => {
                let mut deadlines = BTreeSet::new();
                for (&gateway, row) in &mut table.rows {
                    if row.gone.is_some() {
                        row.gone = Some(now);
                    }
                    row.deadline = self.deadline(row, now);
                    if let Some(deadline) = row.deadline {
                        deadlines.insert((deadline, gateway));
                    }
                }
                table.deadlines = deadlines;
                table.led = term;
            }
            (None, _) => table.led = None,
        }
        term.is_some()
    }

    /// Whether the row is over at `now`: its grace period past with no
    /// session a newer one replaced still running, or twice the period
    /// past whatever runs. A replaced session is told at its replacement
    /// and ends at its next turn; one that never ends holds its gateway's
    /// row for one period more and no longer.
    fn over(&self, row: &Row, now: Instant) -> bool {
        row.gone.is_some_and(|gone| {
            let silent = now.saturating_duration_since(gone);
            let runs = || {
                row.replaced
                    .iter()
                    .any(|session| session.strong_count() > 0)
            };
            silent >= self.grace.saturating_mul(2) || (silent >= self.grace && !runs())
        })
    }

    /// The end of presences, under the table's lock: what the store holds
    /// for each gateway (its lease records and its vetoes) is let go
    /// through the store's one door for a gone gateway, in one write for
    /// them all, then the rows go and their decide streams are told.
    fn end(&self, gone: Vec<(GatewayId, Row)>) {
        let gateways: Vec<GatewayId> = gone.iter().map(|(gateway, _)| *gateway).collect();
        let leases = self.store.drop_gateways(&gateways);
        for ((_, row), leases) in gone.iter().zip(leases) {
            info!(
                gateway = %row.subject,
                leases,
                vetoed = row.vetoes.len(),
                "a gateway's grace period ended with no new Hello: its leases and its vetoes are released"
            );
        }
        drop(gone);
    }

    /// The session `session` of `gateway` ended at `now`: its grace period
    /// starts, unless a newer session stands or the period already runs.
    fn gone(&self, gateway: GatewayId, session: u64, superseded: &Arc<Replaced>, now: Instant) {
        let mut table = self.table.lock();
        let Some(deadline) = table.rows.get_mut(&gateway).map(|row| {
            // The seat releases its Arc after this call. Removing its weak
            // entry here makes its end and the new deadline one locked step.
            row.replaced
                .retain(|replaced| !std::ptr::eq(replaced.as_ptr(), Arc::as_ptr(superseded)));
            if row.session == session && row.gone.is_none() {
                row.gone = Some(now);
            }
            self.deadline(row, now)
        }) else {
            return;
        };
        Self::set_deadline(&mut table, gateway, deadline);
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/service/controller/gateways/tests.rs"]
pub(crate) mod tests;
