//! Who is present (design 4.3, the stream rule): a gateway is present from
//! its `Welcome` until its grace period ends with no new `Hello`, one
//! period more at most while a session a newer one replaced still runs,
//! and what the store holds under its key lives exactly that long: its
//! lease records, its vetoes and its completions.
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

    /// The session ended at `now`, where a case names the instant.
    #[cfg(test)]
    pub(crate) fn end(self, now: Instant) {
        self.gateways
            .gone(self.gateway, self.session, &self.superseded, now);
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

    /// The ceiling of rows.
    #[cfg(test)]
    pub(crate) fn ceiling(&self) -> usize {
        self.ceiling
    }

    /// How many replaced sessions the row of `gateway` notes.
    #[cfg(test)]
    pub(crate) fn noted(&self, gateway: GatewayId) -> usize {
        self.table
            .lock()
            .rows
            .get(&gateway)
            .map_or(0, |row| row.replaced.len())
    }

    /// How many rows the sweep has visited.
    #[cfg(test)]
    fn take_sweep_visits(&self) -> usize {
        std::mem::take(&mut self.table.lock().sweep_visits)
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

    /// The gateway's presence at `now` for a decide stream's open: `None`
    /// when it is not present. No decision asks: a stream's open and end do.
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
    /// a job issues nothing, so nothing else would end its due leases.
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

    /// How many ends the table keeps for the gateway: the row's count,
    /// which every write keeps in step with the note.
    #[cfg(test)]
    pub(crate) fn kept(&self, gateway: GatewayId) -> usize {
        let table = self.table.lock();
        table.rows.get(&gateway).map_or(0, |row| {
            let noted: usize = row.vetoes.values().map(Vec::len).sum();
            assert_eq!(row.held, noted, "the count follows the note");
            let earliest = row.vetoes.values().flatten().map(|&(_, end)| end).min();
            assert!(
                earliest.is_none_or(|end| row.next_end.is_some_and(|next| next <= end)),
                "no end is earlier than the note says"
            );
            row.held
        })
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
    /// for each gateway (its lease records, its vetoes and its completions)
    /// is let go through the store's one door for a gone gateway, in one
    /// write for them all, then the rows go and their decide streams are
    /// told.
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
pub(crate) mod tests {
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };

    use tokio::{sync::watch, task::JoinSet, time::timeout};

    use super::{
        over_kinds, DropPause, Gateways, Refused, SessionPlace, Vetoed, KINDS_MAX, VETOES_MAX,
    };
    use crate::{
        document::spec::{AppliedSpec, PolicySpec},
        engine::{
            sticky_sessions::SessionVerdict,
            view::{Role, Session, WorkerId},
        },
        service::{
            controller::lifecycle::{
                self,
                tests::{a_completed, worker},
            },
            elector::{ElectionSettings, Elector, TickOutcome},
        },
        store::{
            GatewayId, HealthSink, LeaseRecord, LeaseState, LeasesStore, MemoryStore, PickRecord,
            WorkersStore,
        },
        tracking::lease::{Lease, LeaseId},
    };

    pub(crate) const GRACE: Duration = Duration::from_secs(10);
    const A: GatewayId = GatewayId(11);
    const B: GatewayId = GatewayId(12);
    const C: GatewayId = GatewayId(13);
    const D: GatewayId = GatewayId(14);
    const MS: Duration = Duration::from_millis(1);
    /// The term of the first take.
    const TERM: u64 = 1;

    /// Under these a fence taken once stays open for half an hour of a
    /// case's clock: the case's instants decide what is over, not the
    /// election's.
    pub(crate) const PATIENT: ElectionSettings = ElectionSettings {
        duration: Duration::from_secs(3600),
        renew_period: Duration::from_secs(1),
        renew_deadline: Duration::from_secs(1800),
    };

    /// A candidate that leads under its own lease from the returned instant,
    /// a store, and the table over that candidate's fence and that store.
    pub(crate) fn leading(grace: Duration) -> (Elector, Arc<MemoryStore>, Arc<Gateways>, Instant) {
        ceiled(grace, 1_024)
    }

    /// The same, with a table that holds `ceiling` rows.
    fn ceiled(
        grace: Duration,
        ceiling: usize,
    ) -> (Elector, Arc<MemoryStore>, Arc<Gateways>, Instant) {
        let mut elector = Elector::alone("controller-0@10.0.0.7:30100", PATIENT)
            .expect("the settings are in order");
        let taken = Instant::now();
        elector.lead_at(taken);
        assert_eq!(elector.fence().leader_term(taken), Some(1));
        let store = Arc::new(MemoryStore::new());
        let gateways = Arc::new(Gateways::new(
            grace,
            ceiling,
            elector.fence(),
            Arc::clone(&store),
        ));
        (elector, store, gateways, taken)
    }

    /// A `Hello` the table takes.
    fn welcomed(
        gateways: &Arc<Gateways>,
        gateway: GatewayId,
        subject: &str,
        term: u64,
        now: Instant,
    ) -> SessionPlace {
        gateways
            .welcome(gateway, subject, term, now)
            .expect("the table takes the Hello")
    }

    /// One lease of `gateway` as a decision records it.
    pub(crate) fn lease(gateway: GatewayId, id: u64, now: Instant) -> LeaseRecord {
        LeaseRecord {
            lease: Lease {
                id: LeaseId(id),
                ..Lease::default()
            },
            gateway,
            request: id,
            target: WorkerId(7),
            picked: PickRecord::default(),
            decode: None,
            untold: false,
            term: 0,
            issued_at: now,
            dispatch_deadline: now + Duration::from_secs(5),
            state: LeaseState::ISSUED,
        }
    }

    #[test]
    fn a_gateway_is_present_from_its_welcome_until_its_grace_period_ends() {
        let (_elector, _store, gateways, t0) = leading(GRACE);
        assert!(gateways.presence(A, t0).is_none(), "never welcomed");

        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        assert!(gateways.presence(A, t0).is_some());
        // A session that stands is present however long it stands.
        assert!(gateways.presence(A, t0 + 100 * GRACE).is_some());
        assert!(gateways.presence(B, t0).is_none(), "another gateway");

        // Its end starts the period; the clock alone ends the presence, at
        // the period's last instant, with no sweep passed.
        let ended = t0 + Duration::from_secs(60);
        seat.end(ended);
        assert!(gateways.presence(A, ended + GRACE - MS).is_some());
        assert!(gateways.presence(A, ended + GRACE).is_none());
    }

    #[test]
    fn a_hello_inside_the_grace_period_changes_nothing() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        store.issue(lease(A, 1, t0)).expect("a lease");
        welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);

        // The session comes back a millisecond before the period's end.
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0 + GRACE - MS);
        assert_eq!(gateways.sweep(t0 + GRACE), 0);
        assert_eq!(gateways.sweep(t0 + 100 * GRACE), 0);
        assert_eq!(store.live(A), 1, "the gap's lease stands");
        assert!(gateways.presence(A, t0 + 100 * GRACE).is_some());
        drop(seat);
    }

    #[test]
    fn the_sweep_visits_only_rows_with_a_due_deadline() {
        const ROWS: u64 = 4_096;

        let (_elector, store, gateways, t0) = ceiled(GRACE, ROWS as usize);
        let gone = welcomed(&gateways, GatewayId(1), "gateway-1", TERM, t0);
        let standing: Vec<SessionPlace> = (2..=ROWS)
            .map(|id| welcomed(&gateways, GatewayId(id), &format!("gateway-{id}"), TERM, t0))
            .collect();

        assert_eq!(gateways.sweep(t0 + Duration::from_secs(1)), 0);
        assert_eq!(gateways.take_sweep_visits(), 0, "no row has work due");

        let noted = worker(&store, "http://noted:8000");
        assert_eq!(
            standing[0].veto(noted, 1, Some(t0 + 2 * GRACE), t0),
            Vetoed::Shown
        );
        gone.end(t0);
        assert_eq!(gateways.sweep(t0 + GRACE), 1);
        assert_eq!(
            gateways.take_sweep_visits(),
            1,
            "only the expired row is visited"
        );

        assert_eq!(gateways.sweep(t0 + 2 * GRACE), 0);
        assert_eq!(gateways.take_sweep_visits(), 1, "the due note is visited");
        assert_eq!(gateways.kept(GatewayId(2)), 0);
    }

    #[tokio::test]
    async fn past_the_grace_period_the_sweep_lets_the_gateway_go_and_no_other() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        store.issue(lease(A, 1, t0)).expect("a lease");
        store.issue(lease(A, 2, t0)).expect("a lease");
        store.issue(lease(B, 1, t0)).expect("a lease");
        let gone = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let _stands = welcomed(&gateways, B, "gateway-b", TERM, t0);
        let mut of_a = gateways.presence(A, t0).expect("present");
        let mut of_b = gateways.presence(B, t0).expect("present");
        gone.end(t0);

        assert_eq!(gateways.sweep(t0 + GRACE - MS), 0);
        assert_eq!(store.live(A), 2, "inside the period nothing is let go");
        assert_eq!(gateways.sweep(t0 + GRACE), 1);
        assert_eq!(store.live(A), 0, "the gateway's leases are released");
        assert_eq!(store.live(B), 1, "another gateway's stand");
        assert_eq!(gateways.sweep(t0 + GRACE), 0, "let go once");

        // The handle a decide stream holds resolves at the end, and only
        // the ended gateway's.
        timeout(Duration::from_secs(20), of_a.ended())
            .await
            .expect("the ended presence resolves its handle");
        assert!(
            timeout(Duration::from_millis(50), of_b.ended())
                .await
                .is_err(),
            "a presence that stands resolves no handle"
        );
    }

    /// What the store holds under a gateway's key ends with its presence:
    /// its completions with its lease records, and no other gateway's.
    #[test]
    fn a_gateway_let_go_leaves_no_completion_in_the_store() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let served = worker(&store, "http://a:8000");
        let gone = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let _stands = welcomed(&gateways, B, "gateway-b", TERM, t0);
        for gateway in [A, B] {
            lifecycle::completed(
                &store,
                &lifecycle::tests::open(&store, gateway),
                &a_completed(1, served),
                t0,
            )
            .expect("the frame reads");
        }
        gone.end(t0);

        assert_eq!(gateways.sweep(t0 + GRACE - MS), 0);
        assert!(
            store.completed_ok_since(A, served, t0),
            "inside the period nothing is let go"
        );
        assert_eq!(gateways.sweep(t0 + GRACE), 1);
        assert!(
            !store.completed_ok_since(A, served, t0),
            "the gateway's completion went with it"
        );
        assert!(
            store.completed_ok_since(B, served, t0),
            "another gateway's stands"
        );
    }

    /// A `Hello` past the grace period lets the gateway go before it is
    /// welcomed anew, with no pass of the sweep between: the new presence
    /// starts with no completion of the one that ended.
    #[test]
    fn a_gateway_welcomed_anew_past_its_grace_period_starts_with_no_completion() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let served = worker(&store, "http://a:8000");
        let gone = welcomed(&gateways, A, "gateway-a", TERM, t0);
        lifecycle::completed(
            &store,
            &lifecycle::tests::open(&store, A),
            &a_completed(1, served),
            t0,
        )
        .expect("the frame reads");
        gone.end(t0);

        let _anew = welcomed(&gateways, A, "gateway-a", TERM, t0 + GRACE);
        assert!(
            !store.completed_ok_since(A, served, t0),
            "the completion ended with the presence that wrote it"
        );
        assert_eq!(store.health_entries(), 0);
    }

    /// A frame of a replaced session that is past the seat's question may
    /// still be written. The gateway is not let go while that session
    /// runs: what the frame writes lands under the row it was read for and
    /// leaves with it.
    #[test]
    fn a_gateway_is_not_let_go_while_a_session_it_replaced_still_runs() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let served = worker(&store, "http://a:8000");
        let first = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let second = welcomed(&gateways, A, "gateway-a", TERM, t0);
        welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);
        assert!(first.replaced() && second.replaced());
        assert_eq!(gateways.sweep(t0 + GRACE), 0, "two replaced sessions run");
        assert!(gateways.presence(A, t0 + GRACE).is_some());

        // The later of the two ends first: the earlier one holds the row.
        drop(second);
        assert_eq!(gateways.sweep(t0 + GRACE), 0, "a replaced session runs");
        // Its late frame, then its end.
        lifecycle::completed(
            &store,
            &lifecycle::tests::open(&store, A),
            &a_completed(1, served),
            t0 + GRACE,
        )
        .expect("the frame reads");
        drop(first);
        assert_eq!(gateways.sweep(t0 + GRACE), 1);
        assert!(
            !store.completed_ok_since(A, served, t0),
            "the late frame's completion left with the row"
        );
        assert_eq!(store.health_entries(), 0);

        // The other order, on a second gateway: the earlier of the two ends
        // first, and the later one holds the row alone.
        let earlier = welcomed(&gateways, B, "gateway-b", TERM, t0);
        let later = welcomed(&gateways, B, "gateway-b", TERM, t0);
        welcomed(&gateways, B, "gateway-b", TERM, t0).end(t0);
        drop(earlier);
        assert_eq!(
            gateways.sweep(t0 + GRACE),
            0,
            "the later replaced session runs"
        );
        drop(later);
        assert_eq!(gateways.sweep(t0 + GRACE), 1);
    }

    #[test]
    fn a_replaced_sessions_end_and_due_sweep_cleanup_atomically() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        store.issue(lease(A, 1, t0)).expect("a lease");
        let mut replaced = welcomed(&gateways, A, "gateway-a", TERM, t0);
        welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);
        assert!(replaced.replaced());

        let pause = Arc::new(DropPause {
            gone: std::sync::Barrier::new(2),
            swept: std::sync::Barrier::new(2),
        });
        replaced.drop_pause = Some(Arc::clone(&pause));
        let ending = std::thread::spawn(move || drop(replaced));
        pause.gone.wait();
        let let_go = gateways.sweep(t0 + GRACE);
        pause.swept.wait();
        ending.join().expect("the session ends");

        assert_eq!(let_go, 1, "the next due sweep lets the gateway go");
        assert_eq!(store.live(A), 0, "the gateway's lease is released");
    }

    /// A replaced session that never ends holds its gateway's row for one
    /// period more and no longer: the gateway is let go two periods after
    /// its last session ended, whatever still runs.
    #[test]
    fn a_replaced_session_that_never_ends_holds_its_gateway_one_period_more() {
        let (_elector, _store, gateways, t0) = leading(GRACE);
        let never_ends = welcomed(&gateways, A, "gateway-a", TERM, t0);
        welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);
        assert_eq!(gateways.sweep(t0 + 2 * GRACE - MS), 0, "one period more");
        assert!(gateways.presence(A, t0 + 2 * GRACE - MS).is_some());
        assert_eq!(gateways.sweep(t0 + 2 * GRACE), 1, "and no longer");
        assert!(gateways.presence(A, t0 + 2 * GRACE).is_none());
        drop(never_ends);
    }

    /// A row notes only the replaced sessions that still run: a gateway
    /// that says Hello again and again grows no row.
    #[test]
    fn a_row_keeps_no_note_of_a_replaced_session_that_ended() {
        let (_elector, _store, gateways, t0) = leading(GRACE);
        let mut stands = welcomed(&gateways, A, "gateway-a", TERM, t0);
        for _ in 0..8 {
            let newer = welcomed(&gateways, A, "gateway-a", TERM, t0);
            // The replaced session ends on the word.
            drop(stands);
            stands = newer;
        }
        assert_eq!(gateways.noted(A), 0, "no ended session remains");
        drop(stands);
    }

    /// The completions on file are bounded by who is present, never by who
    /// was ever welcomed: waves of gateways under the table's ceiling, each
    /// reporting a completion and let go, leave nothing behind.
    #[test]
    fn the_completions_on_file_do_not_grow_with_the_gateways_ever_welcomed() {
        let (_elector, store, gateways, t0) = ceiled(GRACE, 8);
        let served = worker(&store, "http://a:8000");
        let mut now = t0;
        for wave in 0..5_u64 {
            for n in 0..8_u64 {
                let gateway = GatewayId(100 + wave * 8 + n);
                let subject = format!("gateway-{wave}-{n}");
                let seat = welcomed(&gateways, gateway, &subject, TERM, now);
                lifecycle::completed(
                    &store,
                    &lifecycle::tests::open(&store, gateway),
                    &a_completed(1, served),
                    now,
                )
                .expect("the frame reads");
                seat.end(now);
            }
            assert_eq!(store.health_entries(), 8, "a completion a gateway");
            now += GRACE;
            assert_eq!(gateways.sweep(now), 8);
            assert_eq!(store.health_entries(), 0, "wave {wave}");
        }
    }

    /// A welcome and a letting-go of one gateway at one instant are
    /// ordered: welcomed with what it held, or let go and welcomed with
    /// nothing. Never welcomed and then let go.
    #[tokio::test]
    async fn a_hello_at_the_periods_end_meets_the_letting_go_in_one_order() {
        // The Hello first: the period was over, so the gateway is let go
        // before it is welcomed, and a sweep of that instant lets nothing
        // of the new presence go.
        let (_elector, store, gateways, t0) = leading(GRACE);
        store.issue(lease(A, 1, t0)).expect("a lease");
        welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);
        let mut earlier = gateways.presence(A, t0).expect("present");
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0 + GRACE);
        assert_eq!(store.live(A), 0, "the earlier presence's lease is released");
        timeout(Duration::from_secs(20), earlier.ended())
            .await
            .expect("the earlier presence ended");
        store.issue(lease(A, 2, t0 + GRACE)).expect("a lease");
        assert_eq!(gateways.sweep(t0 + GRACE), 0);
        assert_eq!(gateways.sweep(t0 + 100 * GRACE), 0);
        assert_eq!(store.live(A), 1, "the new presence keeps what it holds");
        drop(seat);

        // The sweep first: the gateway is let go, then welcomed with nothing.
        let (_elector, store, gateways, t0) = leading(GRACE);
        store.issue(lease(A, 1, t0)).expect("a lease");
        welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);
        assert_eq!(gateways.sweep(t0 + GRACE), 1);
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0 + GRACE);
        assert_eq!(store.live(A), 0);
        assert_eq!(gateways.sweep(t0 + GRACE), 0);
        assert!(gateways.presence(A, t0 + GRACE).is_some());
        drop(seat);
    }

    #[test]
    fn a_closed_fence_lets_nothing_go_and_a_take_starts_every_period_again() {
        let (mut elector, store, gateways, t0) = leading(GRACE);
        store.issue(lease(A, 1, t0)).expect("a lease");
        welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);

        // The lease is given up: nobody leads here, and the period's end
        // lets nothing go.
        assert!(elector.release().expect("the release"));
        assert_eq!(gateways.sweep(t0 + GRACE), 0);
        assert_eq!(gateways.sweep(t0 + 3 * GRACE), 0);
        assert_eq!(store.live(A), 1, "a fenced replica releases nothing");

        // The take, three periods later: the gateway could say no Hello
        // here meanwhile, so its period starts at the take.
        let retaken = t0 + 3 * GRACE;
        assert_eq!(
            elector.tick(retaken).expect("the take"),
            TickOutcome::Acquired { term: 2 }
        );
        assert_eq!(gateways.sweep(retaken), 0);
        assert!(gateways.presence(A, retaken + GRACE - MS).is_some());
        assert_eq!(gateways.sweep(retaken + GRACE - MS), 0);
        assert_eq!(store.live(A), 1);
        assert_eq!(gateways.sweep(retaken + GRACE), 1);
        assert_eq!(store.live(A), 0);
    }

    /// A renewal that lands late opens the fence again under the same term.
    /// No `Hello` could be welcomed here while it was closed, so the first
    /// read after it starts every gone gateway's period again, as a take
    /// does.
    #[test]
    fn a_late_renewal_under_the_same_term_starts_every_gone_gateways_period_again() {
        let (mut elector, store, gateways, t0) = leading(GRACE);
        store.issue(lease(A, 1, t0)).expect("a lease");
        welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);

        // Past the renew deadline the fence is closed and the lease is
        // still this candidate's: a read finds nobody leading here.
        let closed = t0 + PATIENT.renew_deadline + MS;
        assert_eq!(gateways.sweep(closed), 0);
        assert_eq!(store.live(A), 1, "a fenced replica releases nothing");

        // The renewal lands late, under the same term: the gateway's
        // period starts at the first read after it.
        let landed = closed + GRACE;
        assert_eq!(
            elector.tick(landed).expect("the renewal"),
            TickOutcome::Renewed { term: TERM }
        );
        assert_eq!(gateways.sweep(landed), 0);
        assert!(gateways.presence(A, landed + GRACE - MS).is_some());
        assert_eq!(gateways.sweep(landed + GRACE - MS), 0);
        assert_eq!(store.live(A), 1);
        assert_eq!(gateways.sweep(landed + GRACE), 1);
        assert_eq!(store.live(A), 0);
    }

    /// The take's rule holds at a Hello too: a Hello that is the first read
    /// after a take finds the period started again, and lets nothing go.
    #[test]
    fn a_hello_right_after_a_take_keeps_what_the_gateway_held() {
        let (mut elector, store, gateways, t0) = leading(GRACE);
        store.issue(lease(A, 1, t0)).expect("a lease");
        welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);
        assert!(elector.release().expect("the release"));
        assert_eq!(gateways.sweep(t0 + GRACE), 0);

        let retaken = t0 + 3 * GRACE;
        assert_eq!(
            elector.tick(retaken).expect("the take"),
            TickOutcome::Acquired { term: 2 }
        );
        let seat = welcomed(&gateways, A, "gateway-a", 2, retaken);
        assert_eq!(store.live(A), 1, "its lease stands through the gap");
        drop(seat);
    }

    #[tokio::test]
    async fn a_second_session_supersedes_the_first_whose_end_starts_no_period() {
        let (_elector, _store, gateways, t0) = leading(GRACE);
        let first = welcomed(&gateways, A, "gateway-a", TERM, t0);
        assert!(
            timeout(Duration::from_millis(50), first.superseded())
                .await
                .is_err(),
            "a lone session is not superseded"
        );
        let second = welcomed(&gateways, A, "gateway-a", TERM, t0 + MS);
        timeout(Duration::from_secs(20), first.superseded())
            .await
            .expect("the first session is told");
        assert!(
            timeout(Duration::from_millis(50), second.superseded())
                .await
                .is_err(),
            "the newer session stands"
        );

        // The superseded stream's end is not the gateway's.
        first.end(t0 + 2 * MS);
        assert!(gateways.presence(A, t0 + 100 * GRACE).is_some());
        assert_eq!(gateways.sweep(t0 + 100 * GRACE), 0);
        // The standing one's is.
        second.end(t0 + 3 * MS);
        assert!(gateways.presence(A, t0 + 3 * MS + GRACE).is_none());
    }

    #[test]
    fn a_seats_drop_is_its_sessions_end() {
        let (_elector, _store, gateways, t0) = leading(GRACE);
        drop(welcomed(&gateways, A, "gateway-a", TERM, t0));
        let dropped = Instant::now();
        assert!(gateways.presence(A, dropped).is_some());
        assert!(gateways.presence(A, dropped + GRACE).is_none());
    }

    /// A session's number is its alone for the table's life: the late end
    /// of a session that an earlier presence replaced is no end of the
    /// session that stands now. A replaced session reaches a later
    /// presence only by outliving two periods.
    #[test]
    fn a_replaced_sessions_late_end_is_no_end_of_a_later_presences_session() {
        let (_elector, _store, gateways, t0) = leading(GRACE);
        let first = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let second = welcomed(&gateways, A, "gateway-a", TERM, t0 + MS);
        second.end(t0 + 2 * MS);
        assert_eq!(gateways.sweep(t0 + 2 * MS + 2 * GRACE), 1);

        // A new presence, and only then the end of the replaced session.
        let third = welcomed(&gateways, A, "gateway-a", TERM, t0 + 2 * GRACE + 3 * MS);
        first.end(t0 + 2 * GRACE + 4 * MS);
        let later = t0 + 5 * GRACE;
        assert!(gateways.presence(A, later).is_some(), "its session stands");
        assert_eq!(gateways.sweep(later), 0);
        drop(third);
    }

    #[test]
    fn the_session_counter_never_wraps_or_reuses_zero() {
        let (_elector, _store, gateways, t0) = leading(GRACE);
        gateways.table.lock().sessions = u64::MAX - 1;
        let stands = welcomed(&gateways, A, "gateway-a", TERM, t0);
        assert_eq!(stands.session, u64::MAX);

        let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gateways.welcome(A, "gateway-a", TERM, t0)
        }));
        assert!(attempted.is_ok(), "counter exhaustion must not panic");
        assert!(
            matches!(attempted.expect("checked above"), Err(Refused::Exhausted)),
            "an exhausted counter refuses the Hello"
        );
        assert!(!stands.replaced(), "the standing session is not superseded");
        let table = gateways.table.lock();
        let row = table.rows.get(&A).expect("the standing row");
        assert_eq!(table.sessions, u64::MAX);
        assert_eq!(row.session, u64::MAX);
        assert!(row.gone.is_none(), "the row's session still stands");
    }

    /// The seat is the table's own read of the fence: a replica that
    /// stopped leading after the session's read seats nobody and releases
    /// nothing.
    #[test]
    fn a_hello_whose_fence_closed_since_the_sessions_read_takes_no_seat_and_lets_nothing_go() {
        let (mut elector, store, gateways, t0) = leading(GRACE);
        store.issue(lease(A, 1, t0)).expect("a lease");
        welcomed(&gateways, A, "gateway-a", TERM, t0).end(t0);

        // The period is over with no sweep passed, and the lease is given
        // up between a session's read and the table's lock.
        assert!(elector.release().expect("the release"));
        assert_eq!(
            gateways.welcome(A, "gateway-a", TERM, t0 + GRACE).err(),
            Some(Refused::Stepped {
                read: TERM,
                shown: None
            })
        );
        assert_eq!(store.live(A), 1, "a fenced replica releases nothing");
    }

    /// A `Hello` read under a term the election has left takes no seat:
    /// the session that stands under the later term keeps its own.
    #[tokio::test]
    async fn a_hello_read_under_an_earlier_term_supersedes_no_session_of_the_later_one() {
        let (mut elector, _store, gateways, t0) = leading(GRACE);
        // The replica gives its lease up and takes it again: term 2.
        assert!(elector.release().expect("the release"));
        let again = t0 + MS;
        assert_eq!(
            elector.tick(again).expect("the take"),
            TickOutcome::Acquired { term: 2 }
        );
        let stands = welcomed(&gateways, A, "gateway-a", 2, again);

        // A Hello its session read under term 1 reaches the table now.
        assert_eq!(
            gateways.welcome(A, "gateway-a", TERM, again).err(),
            Some(Refused::Stepped {
                read: TERM,
                shown: Some(2)
            })
        );
        assert!(
            timeout(Duration::from_millis(50), stands.superseded())
                .await
                .is_err(),
            "the standing session keeps its seat"
        );
        assert!(gateways.presence(A, again).is_some());
        drop(stands);
    }

    /// The table's bound that is not time: a `Hello` that would make a row
    /// past the ceiling is refused at the door, and a present gateway's is
    /// always taken.
    #[test]
    fn a_hello_past_the_ceiling_is_refused_and_a_present_gateways_is_taken() {
        let (_elector, _store, gateways, t0) = ceiled(GRACE, 3);
        let first = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let second = welcomed(&gateways, B, "gateway-b", TERM, t0);
        let third = welcomed(&gateways, C, "gateway-c", TERM, t0);
        // The fourth would be a new row: refused, and no row is made.
        assert_eq!(
            gateways.welcome(D, "gateway-d", TERM, t0).err(),
            Some(Refused::Full { ceiling: 3 })
        );
        assert!(gateways.presence(D, t0).is_none());
        // A present gateway says Hello again: no new row, so it is taken.
        let again = welcomed(&gateways, A, "gateway-a", TERM, t0);

        // Inside its grace period a gateway's row still counts.
        third.end(t0);
        assert_eq!(
            gateways
                .welcome(D, "gateway-d", TERM, t0 + GRACE - MS)
                .err(),
            Some(Refused::Full { ceiling: 3 })
        );
        // Once the sweep let one go, the fourth is welcomed.
        assert_eq!(gateways.sweep(t0 + GRACE), 1);
        let fourth = welcomed(&gateways, D, "gateway-d", TERM, t0 + GRACE);
        drop((first, second, again, fourth));
    }

    /// One key, one subject: a `Hello` of another subject under a present
    /// gateway's key is refused, and nothing of that gateway is touched.
    /// The table is handed the key and the subject apart, so the case names
    /// one key under two subjects.
    #[tokio::test]
    async fn a_hello_of_another_subject_under_a_present_gateways_key_is_refused() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        store.issue(lease(A, 1, t0)).expect("a lease");
        let stands = welcomed(&gateways, A, "gateway-a", TERM, t0);
        assert_eq!(
            gateways.welcome(A, "gateway-b", TERM, t0).err(),
            Some(Refused::AnotherSubject)
        );
        // The standing session is not replaced, and what it held stands.
        assert!(
            timeout(Duration::from_millis(50), stands.superseded())
                .await
                .is_err(),
            "the present gateway's session stands"
        );
        assert_eq!(store.live(A), 1);

        // Inside the grace period the key is still the first subject's.
        stands.end(t0);
        assert_eq!(
            gateways
                .welcome(A, "gateway-b", TERM, t0 + GRACE - MS)
                .err(),
            Some(Refused::AnotherSubject)
        );
        // Once the first is let go the key is free, and the other subject
        // starts with nothing of it.
        let other = welcomed(&gateways, A, "gateway-b", TERM, t0 + GRACE);
        assert_eq!(store.live(A), 0, "the first gateway's lease went with it");
        drop(other);
    }

    /// The task over the real clock: a gone gateway is let go a grace
    /// period after its session's end, and the task ends at its signal.
    #[tokio::test]
    async fn the_sweeps_task_lets_a_gone_gateway_go_and_ends_at_its_stop() {
        let grace = Duration::from_millis(60);
        let (_elector, store, gateways, t0) = leading(grace);
        store.issue(lease(A, 1, t0)).expect("a lease");
        let (stop, stopped) = watch::channel(false);
        let mut task = JoinSet::new();
        let sweeper = Arc::clone(&gateways);
        task.spawn(async move { sweeper.run(stopped).await });

        let ended = Instant::now();
        drop(welcomed(&gateways, A, "gateway-a", TERM, ended));
        let started = Instant::now();
        while store.live(A) > 0 {
            assert!(
                started.elapsed() < Duration::from_secs(20),
                "the sweep let nothing go in 20 s"
            );
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        assert!(
            ended.elapsed() >= grace,
            "let go {:?} after the end, inside the period of {grace:?}",
            ended.elapsed()
        );

        stop.send_replace(true);
        timeout(Duration::from_secs(20), task.join_next())
            .await
            .expect("the task ends within 20 s of its stop")
            .expect("one task")
            .expect("it ended by itself");
    }

    /// The leases' due ends ride the sweeps' task: a lease past its
    /// dispatch deadline leaves by the task's own tick where no decider
    /// issues, its gateway's session standing, and a lease inside its
    /// deadline stays.
    #[tokio::test]
    async fn the_sweeps_task_ends_a_lease_past_its_dispatch_deadline() {
        let (_elector, store, gateways, t0) = leading(Duration::from_millis(60));
        let due = LeaseRecord {
            dispatch_deadline: t0,
            ..lease(A, 1, t0)
        };
        store.issue(due).expect("a lease");
        store.issue(lease(A, 2, t0)).expect("a lease");
        let _stands = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let (stop, stopped) = watch::channel(false);
        let mut task = JoinSet::new();
        let sweeper = Arc::clone(&gateways);
        task.spawn(async move { sweeper.run(stopped).await });

        let started = Instant::now();
        while store.live(A) > 1 {
            assert!(
                started.elapsed() < Duration::from_secs(20),
                "the sweep ended no due lease in 20 s"
            );
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        assert!(LeasesStore::get(&*store, A, LeaseId(1)).is_none());
        assert!(LeasesStore::get(&*store, A, LeaseId(2)).is_some());

        stop.send_replace(true);
        timeout(Duration::from_secs(20), task.join_next())
            .await
            .expect("the task ends within 20 s of its stop")
            .expect("one task")
            .expect("it ended by itself");
    }

    /// The names that left ride the sweeps' task: one that stays away past
    /// the absence's time takes its session pins with it by the task's own
    /// tick, which no other caller of the store's tick stands in for.
    #[tokio::test]
    async fn the_sweeps_task_ends_the_absence_of_a_key_that_stays_away() {
        let (_elector, store, gateways, _t0) = leading(Duration::from_millis(60));
        // The absence's time is the longest TTL of the document's session
        // rules: one second.
        let mut document = PolicySpec::preset("manual").expect("a preset of this build");
        document.generation = 1;
        let rule = document.bindings[0].sessions.as_mut();
        let rule = rule.expect("the preset's rule");
        (rule.ttl_s, rule.sweep_s) = (1, 1);
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        store.apply(spec).expect("a first generation");
        let w = worker(&store, "http://a:8000");
        let key = Session::from_raw(7);
        let held = || {
            let world = store.snapshot();
            let table = world.spec().engine().sessions();
            table.pins(key).map(|pins| pins[0])
        };
        let world = store.snapshot();
        let table = world.spec().engine().sessions();
        table.dispatched(key, SessionVerdict::Pinned, w);
        let revision = WorkersStore::get(&*store, w).expect("the row").revision;
        WorkersStore::remove(&*store, w, revision).expect("removed");
        assert_eq!(held(), Some(w), "inside its time");
        let (stop, stopped) = watch::channel(false);
        let mut task = JoinSet::new();
        let sweeper = Arc::clone(&gateways);
        task.spawn(async move { sweeper.run(stopped).await });
        let started = Instant::now();
        while held().is_some() {
            assert!(
                started.elapsed() < Duration::from_secs(20),
                "the task ended no absence in 20 s"
            );
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        stop.send_replace(true);
        timeout(Duration::from_secs(20), task.join_next())
            .await
            .expect("the task ends within 20 s of its stop")
            .expect("one task")
            .expect("it ended by itself");
    }

    const BREAKER: i32 = 1;
    const LIVENESS: i32 = 2;
    const S: Duration = Duration::from_secs(1);

    #[test]
    fn the_store_shows_the_latest_end_of_a_workers_kinds_and_a_clear_lifts_its_kind_alone() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let w = worker(&store, "http://a:8000");
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        assert_eq!(seat.veto(w, BREAKER, Some(t0 + 5 * S), t0), Vetoed::Shown);
        assert_eq!(seat.veto(w, LIVENESS, Some(t0 + 9 * S), t0), Vetoed::Shown);
        // The later end is shown, to its gateway alone.
        assert!(store.vetoed(A, w, t0 + 9 * S - MS));
        assert!(!store.vetoed(A, w, t0 + 9 * S));
        assert!(!store.vetoed(B, w, t0));

        // The later kind cleared: the earlier end is shown again.
        assert_eq!(seat.veto(w, LIVENESS, None, t0 + S), Vetoed::Shown);
        assert!(store.vetoed(A, w, t0 + 5 * S - MS));
        assert!(!store.vetoed(A, w, t0 + 5 * S));
        // A kind set again moves its own end.
        assert_eq!(
            seat.veto(w, BREAKER, Some(t0 + 3 * S), t0 + S),
            Vetoed::Shown
        );
        assert!(!store.vetoed(A, w, t0 + 3 * S));
        // The last kind cleared lifts the veto, and nothing is kept.
        assert_eq!(seat.veto(w, BREAKER, None, t0 + 2 * S), Vetoed::Shown);
        assert!(!store.vetoed(A, w, t0 + 2 * S));
        assert_eq!(gateways.kept(A), 0);
    }

    /// An end at the frame's own instant is no veto: the frame lifts its
    /// kind as a clear does, and the worker is not read as one with no row.
    #[test]
    fn an_end_at_the_frames_instant_lifts_its_kind_as_a_clear_does() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let w = worker(&store, "http://a:8000");
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        assert_eq!(seat.veto(w, BREAKER, Some(t0 + 5 * S), t0), Vetoed::Shown);
        assert_eq!(seat.veto(w, BREAKER, Some(t0 + S), t0 + S), Vetoed::Shown);
        assert!(!store.vetoed(A, w, t0 + S));
        assert_eq!(gateways.kept(A), 0);
    }

    /// A kind is kept by its number: one this build has no name for stands
    /// and clears like the named ones.
    #[test]
    fn a_kind_this_build_has_no_name_for_stands_and_clears_like_the_others() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let w = worker(&store, "http://a:8000");
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        assert_eq!(seat.veto(w, 7, Some(t0 + 9 * S), t0), Vetoed::Shown);
        assert_eq!(seat.veto(w, BREAKER, Some(t0 + 5 * S), t0), Vetoed::Shown);
        assert_eq!(seat.veto(w, BREAKER, None, t0), Vetoed::Shown);
        assert!(
            store.vetoed(A, w, t0 + 9 * S - MS),
            "the unnamed kind stands"
        );
        assert_eq!(seat.veto(w, 7, None, t0), Vetoed::Shown);
        assert!(!store.vetoed(A, w, t0));
    }

    /// What the published world's veto reader hands `gateway` of the worker
    /// at `now`: whether a hold stands on it, and whether a veto does.
    fn bits(
        store: &MemoryStore,
        gateway: GatewayId,
        worker: WorkerId,
        now: Instant,
    ) -> (bool, bool) {
        // A registration is in force at its window's publication: the case
        // asks for one.
        store.publish_now();
        let world = store.snapshot();
        assert!(world.first_row_of(worker).is_some(), "the worker's row");
        let vetoes = world.gateway_vetoes("m", Role::Single, gateway);
        let veto = vetoes.standing_at(now).find(|veto| veto.id == worker);
        veto.map_or((false, false), |veto| (veto.held, true))
    }

    /// The store reads no kind: the note says which ends hold. Every kind
    /// but the doubt of liveness holds, a number this build has no name for
    /// among them, and the hold shown is the latest of their ends. A worker
    /// reads to its gateway as masked to its veto's end, and as behind an
    /// open breaker too while a hold stands.
    #[test]
    fn the_store_is_shown_the_hold_among_a_workers_kinds() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let w = worker(&store, "http://a:8000");
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let (held, doubted, clear) = ((true, true), (false, true), (false, false));
        // A doubt of liveness alone holds nothing.
        assert_eq!(seat.veto(w, LIVENESS, Some(t0 + 9 * S), t0), Vetoed::Shown);
        assert_eq!(bits(&store, A, w, t0), doubted);
        // A breaker's window beside it holds to its own end, and the doubt
        // stands from there.
        assert_eq!(seat.veto(w, BREAKER, Some(t0 + 5 * S), t0), Vetoed::Shown);
        assert_eq!(bits(&store, A, w, t0 + 5 * S - MS), held);
        assert_eq!(bits(&store, A, w, t0 + 5 * S), doubted);
        assert_eq!(bits(&store, A, w, t0 + 9 * S), clear);
        assert_eq!(bits(&store, B, w, t0), clear);
        // The doubt cleared: the breaker's window holds alone.
        assert_eq!(seat.veto(w, LIVENESS, None, t0), Vetoed::Shown);
        assert_eq!(bits(&store, A, w, t0), held);
        assert_eq!(bits(&store, A, w, t0 + 5 * S), clear);
        // The breaker closed beside a new doubt: nothing holds.
        assert_eq!(seat.veto(w, LIVENESS, Some(t0 + 9 * S), t0), Vetoed::Shown);
        assert_eq!(seat.veto(w, BREAKER, None, t0), Vetoed::Shown);
        assert_eq!(bits(&store, A, w, t0), doubted);
        // A kind with no name holds: the rule lifts what it can name.
        assert_eq!(seat.veto(w, 7, Some(t0 + 3 * S), t0), Vetoed::Shown);
        assert_eq!(bits(&store, A, w, t0), held);
        assert_eq!(bits(&store, A, w, t0 + 3 * S), doubted);
    }

    /// A `Hello`'s list is shown the same way: each listed worker's latest
    /// end, with the hold among its kinds.
    #[test]
    fn a_hellos_list_is_shown_with_the_hold_among_each_workers_kinds() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let (held, doubted) = (
            worker(&store, "http://a:8000"),
            worker(&store, "http://b:8000"),
        );
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let list = [
            (held, BREAKER, t0 + 5 * S),
            (held, LIVENESS, t0 + 9 * S),
            (doubted, LIVENESS, t0 + 9 * S),
        ];
        assert_eq!(seat.restore(&list, t0), 3);
        assert_eq!(bits(&store, A, held, t0), (true, true));
        assert_eq!(bits(&store, A, held, t0 + 5 * S), (false, true));
        assert_eq!(bits(&store, A, doubted, t0), (false, true));
        assert_eq!(bits(&store, A, doubted, t0 + 9 * S), (false, false));
    }

    #[test]
    fn a_veto_on_a_worker_with_no_row_is_dropped_and_nothing_of_it_is_kept() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let unheld = WorkerId(77);
        assert_eq!(
            seat.veto(unheld, BREAKER, Some(t0 + 5 * S), t0),
            Vetoed::NoWorker
        );
        assert!(!store.vetoed(A, unheld, t0));
        assert_eq!(gateways.kept(A), 0);
        // Its clear finds nothing to lift.
        assert_eq!(seat.veto(unheld, BREAKER, None, t0), Vetoed::Shown);
    }

    /// A veto is about a worker's row and leaves with it: a worker removed
    /// and registered again starts clean, whatever the table noted.
    #[test]
    fn a_worker_registered_again_starts_clean_and_an_end_past_its_instant_is_not_kept() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let w = worker(&store, "http://a:8000");
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        assert_eq!(seat.veto(w, BREAKER, Some(t0 + 60 * S), t0), Vetoed::Shown);
        let revision = WorkersStore::get(&*store, w).expect("the row").revision;
        WorkersStore::remove(&*store, w, revision).expect("removed");
        assert_eq!(worker(&store, "http://a:8000"), w, "the same id");
        assert!(!store.vetoed(A, w, t0), "the veto left with the row");

        // The other kind's frame shows its own end, not the removed row's.
        assert_eq!(
            seat.veto(w, LIVENESS, Some(t0 + 5 * S), t0 + S),
            Vetoed::Shown
        );
        assert!(store.vetoed(A, w, t0 + 5 * S - MS));
        assert!(!store.vetoed(A, w, t0 + 5 * S));
        assert_eq!(gateways.kept(A), 1);

        // Past their instants the ends are no vetoes: after a pass the
        // table keeps none, however many workers came and went.
        for n in 0..40 {
            let w = worker(&store, &format!("http://w{n}:8000"));
            assert_eq!(
                seat.veto(w, BREAKER, Some(t0 + 5 * S), t0 + S),
                Vetoed::Shown
            );
        }
        assert_eq!(gateways.kept(A), 41);
        assert_eq!(gateways.sweep(t0 + 5 * S), 0);
        assert_eq!(gateways.kept(A), 0);
    }

    #[test]
    fn the_letting_go_lifts_every_veto_of_the_gateway_and_no_other_gateways() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let w = worker(&store, "http://a:8000");
        let gone = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let stands = welcomed(&gateways, B, "gateway-b", TERM, t0);
        let until = t0 + 100 * GRACE;
        assert_eq!(gone.veto(w, BREAKER, Some(until), t0), Vetoed::Shown);
        assert_eq!(stands.veto(w, BREAKER, Some(until), t0), Vetoed::Shown);
        gone.end(t0);

        // Inside the period the veto stands, as the leases do.
        assert_eq!(gateways.sweep(t0 + GRACE - MS), 0);
        assert!(store.vetoed(A, w, t0 + GRACE - MS));
        assert_eq!(gateways.sweep(t0 + GRACE), 1);
        assert!(!store.vetoed(A, w, t0 + GRACE));
        assert!(store.vetoed(B, w, t0 + GRACE), "another gateway's stands");
        drop(stands);
    }

    /// One pass lets every gateway whose period is over go in one write of
    /// the store: their vetoes leave in one publication, and a present
    /// gateway's stands. A pass that lets none go publishes nothing.
    #[test]
    fn a_pass_lets_its_gateways_go_in_one_publication_of_the_store() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let w = worker(&store, "http://a:8000");
        let until = t0 + 100 * GRACE;
        let stands = welcomed(&gateways, C, "gateway-c", TERM, t0);
        assert_eq!(stands.veto(w, BREAKER, Some(until), t0), Vetoed::Shown);
        for (gateway, subject) in [(A, "gateway-a"), (B, "gateway-b")] {
            let gone = welcomed(&gateways, gateway, subject, TERM, t0);
            assert_eq!(gone.veto(w, BREAKER, Some(until), t0), Vetoed::Shown);
            gone.end(t0);
        }

        let published = store.publications();
        assert_eq!(gateways.sweep(t0 + GRACE - MS), 0);
        assert_eq!(store.publications(), published);
        assert_eq!(gateways.sweep(t0 + GRACE), 2);
        assert_eq!(
            store.publications(),
            published + 1,
            "one world for the pass"
        );
        assert!(!store.vetoed(A, w, t0 + GRACE) && !store.vetoed(B, w, t0 + GRACE));
        assert!(store.vetoed(C, w, t0 + GRACE), "a present gateway's stands");
        drop(stands);
    }

    #[test]
    fn a_replaced_sessions_veto_writes_nothing() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let w = worker(&store, "http://a:8000");
        let first = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let second = welcomed(&gateways, A, "gateway-a", TERM, t0);
        assert_eq!(
            first.veto(w, BREAKER, Some(t0 + 5 * S), t0),
            Vetoed::Replaced
        );
        assert!(!store.vetoed(A, w, t0));
        assert_eq!(second.veto(w, BREAKER, Some(t0 + 5 * S), t0), Vetoed::Shown);
        // It lifts nothing either, and its list restores nothing.
        assert_eq!(first.veto(w, BREAKER, None, t0), Vetoed::Replaced);
        assert_eq!(first.restore(&[], t0), 0);
        assert!(store.vetoed(A, w, t0));
        drop((first, second));
    }

    /// A `Hello`'s list is the gateway's vetoes, whole: what it omits is
    /// lifted, and of two entries of one worker and kind the later end
    /// stands, in either order.
    #[test]
    fn a_hellos_list_is_the_gateways_vetoes_whole_and_the_later_end_of_a_pair_stands() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let listed = worker(&store, "http://a:8000");
        let omitted = worker(&store, "http://b:8000");
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        assert_eq!(
            seat.veto(listed, BREAKER, Some(t0 + 60 * S), t0),
            Vetoed::Shown
        );
        assert_eq!(
            seat.veto(omitted, BREAKER, Some(t0 + 60 * S), t0),
            Vetoed::Shown
        );
        seat.end(t0);

        // Back inside the period. One veto is listed with a nearer end, a
        // second kind twice, one worker is not held, one end is past.
        let now = t0 + S;
        let seat = welcomed(&gateways, A, "gateway-a", TERM, now);
        for pair in [[6, 7], [7, 6]] {
            let list = [
                (listed, BREAKER, now + 3 * S),
                (listed, LIVENESS, now + pair[0] * S),
                (listed, LIVENESS, now + pair[1] * S),
                (WorkerId(77), BREAKER, now + 9 * S),
                (omitted, LIVENESS, now),
            ];
            assert_eq!(seat.restore(&list, now), 2, "two ends of one worker");
            assert!(store.vetoed(A, listed, now + 7 * S - MS));
            assert!(!store.vetoed(A, listed, now + 7 * S), "the later end");
            assert!(!store.vetoed(A, omitted, now), "what the list omits");
            assert_eq!(gateways.kept(A), 2);
        }
        // An empty list lifts everything.
        assert_eq!(seat.restore(&[], now), 0);
        assert!(!store.vetoed(A, listed, now));
        assert_eq!(gateways.kept(A), 0);
        drop(seat);
    }

    /// A worker holds [`KINDS_MAX`] kinds for a gateway and no more: the
    /// veto that would set one more writes nothing, and a kind that stands
    /// or a clear is always taken.
    #[test]
    fn a_worker_holds_its_ceiling_of_kinds_and_one_more_writes_nothing() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let w = worker(&store, "http://a:8000");
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let kinds = i32::try_from(KINDS_MAX).expect("a small number");
        for kind in 1..=kinds {
            assert_eq!(seat.veto(w, kind, Some(t0 + 5 * S), t0), Vetoed::Shown);
        }
        assert_eq!(gateways.kept(A), KINDS_MAX);

        // One more kind: nothing is written, and what stands stays.
        assert_eq!(seat.veto(w, kinds + 1, Some(t0 + 9 * S), t0), Vetoed::Full);
        assert_eq!(gateways.kept(A), KINDS_MAX);
        assert!(store.vetoed(A, w, t0 + 5 * S - MS));
        assert!(
            !store.vetoed(A, w, t0 + 5 * S),
            "the refused end is not shown"
        );

        // A kind that stands moves its own end; a clear of a kind that
        // does not stand is taken; a clear makes room for the new kind.
        assert_eq!(seat.veto(w, 1, Some(t0 + 6 * S), t0), Vetoed::Shown);
        assert_eq!(seat.veto(w, kinds + 1, None, t0), Vetoed::Shown);
        assert_eq!(seat.veto(w, 2, None, t0), Vetoed::Shown);
        assert_eq!(seat.veto(w, kinds + 1, Some(t0 + 9 * S), t0), Vetoed::Shown);
        assert!(store.vetoed(A, w, t0 + 9 * S - MS));
        assert_eq!(gateways.kept(A), KINDS_MAX);
        drop(seat);
    }

    /// A ceiling stands in the way only while the note's ends stand: a
    /// worker removed and registered again holds none, and the kind that
    /// was one too many is taken.
    #[test]
    fn a_workers_ceiling_of_kinds_leaves_with_its_row() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let w = worker(&store, "http://a:8000");
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let kinds = i32::try_from(KINDS_MAX).expect("a small number");
        for kind in 1..=kinds {
            assert_eq!(seat.veto(w, kind, Some(t0 + 60 * S), t0), Vetoed::Shown);
        }
        assert_eq!(seat.veto(w, kinds + 1, Some(t0 + 60 * S), t0), Vetoed::Full);
        let revision = WorkersStore::get(&*store, w).expect("the row").revision;
        WorkersStore::remove(&*store, w, revision).expect("removed");
        assert_eq!(worker(&store, "http://a:8000"), w, "the same id");
        assert_eq!(seat.veto(w, kinds + 1, Some(t0 + 5 * S), t0), Vetoed::Shown);
        assert!(
            !store.vetoed(A, w, t0 + 5 * S),
            "no end of the row before it"
        );
        assert_eq!(gateways.kept(A), 1);
        drop(seat);
    }

    /// A gateway holds [`VETOES_MAX`] vetoes and no more, what its `Hello`
    /// lists whole: the frame that would set one more writes nothing. A
    /// kind that stands and a clear are always taken, and an end past its
    /// instant leaves the count at the frame that needs its place, with no
    /// pass of the sweep between.
    #[test]
    fn a_gateway_holds_its_ceiling_of_vetoes_and_one_more_writes_nothing() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let kinds = i32::try_from(KINDS_MAX).expect("a small number");
        let workers: Vec<WorkerId> = (0..VETOES_MAX / KINDS_MAX)
            .map(|n| worker(&store, &format!("http://w{n}:8000")))
            .collect();
        // The first worker's ends pass at five seconds, the others' at sixty.
        let list: Vec<(WorkerId, i32, Instant)> = workers
            .iter()
            .enumerate()
            .flat_map(|(n, &w)| {
                let end = t0 + if n == 0 { 5 * S } else { 60 * S };
                (1..=kinds).map(move |kind| (w, kind, end))
            })
            .collect();
        assert_eq!(seat.restore(&list, t0), VETOES_MAX);
        assert_eq!(gateways.kept(A), VETOES_MAX);

        // One veto more, on a worker that holds none: nothing is written.
        let one_more = worker(&store, "http://one-more:8000");
        assert_eq!(
            seat.veto(one_more, BREAKER, Some(t0 + 60 * S), t0 + S),
            Vetoed::Over
        );
        assert!(!store.vetoed(A, one_more, t0 + S));
        assert_eq!(gateways.kept(A), VETOES_MAX);
        // A kind that stands moves its own end, and a clear is taken.
        assert_eq!(
            seat.veto(workers[1], 1, Some(t0 + 61 * S), t0 + S),
            Vetoed::Shown
        );
        assert_eq!(seat.veto(one_more, BREAKER, None, t0 + S), Vetoed::Shown);
        assert_eq!(gateways.kept(A), VETOES_MAX);

        // Past the first worker's instant its eight ends are no vetoes:
        // the frame takes a place they held.
        assert_eq!(
            seat.veto(one_more, BREAKER, Some(t0 + 60 * S), t0 + 5 * S),
            Vetoed::Shown
        );
        assert!(store.vetoed(A, one_more, t0 + 5 * S));
        assert_eq!(gateways.kept(A), VETOES_MAX - KINDS_MAX + 1);
        drop(seat);
    }

    /// A list that fills a note to its ceiling: [`KINDS_MAX`] kinds on each
    /// of `workers`, every end at sixty seconds.
    fn full_list(workers: &[WorkerId], t0: Instant) -> Vec<(WorkerId, i32, Instant)> {
        let kinds = i32::try_from(KINDS_MAX).expect("a small number");
        workers
            .iter()
            .flat_map(|&w| (1..=kinds).map(move |kind| (w, kind, t0 + 60 * S)))
            .collect()
    }

    /// Fills the seat's note to its ceiling of [`VETOES_MAX`] ends on newly
    /// registered workers ([`full_list`]). Returns the workers.
    fn filled(store: &MemoryStore, seat: &SessionPlace, t0: Instant) -> Vec<WorkerId> {
        let workers: Vec<WorkerId> = (0..VETOES_MAX / KINDS_MAX)
            .map(|n| worker(store, &format!("http://w{n}:8000")))
            .collect();
        assert_eq!(seat.restore(&full_list(&workers, t0), t0), VETOES_MAX);
        workers
    }

    /// A gateway's ceiling counts the vetoes the store shows: the ends of
    /// workers whose rows left go before the refusal, and the frame that
    /// would set one more is taken.
    #[test]
    fn a_gateways_ceiling_counts_no_end_of_a_row_that_left() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let workers = filled(&store, &seat, t0);
        // Every row but the first leaves, and its vetoes with it: the note
        // still holds their ends.
        for &w in &workers[1..] {
            let revision = WorkersStore::get(&*store, w).expect("the row").revision;
            WorkersStore::remove(&*store, w, revision).expect("removed");
        }
        assert_eq!(gateways.kept(A), VETOES_MAX);

        // One veto more, on a worker that holds none: the ends of the rows
        // that left go, and the veto is shown.
        let one_more = worker(&store, "http://one-more:8000");
        assert_eq!(
            seat.veto(one_more, BREAKER, Some(t0 + 60 * S), t0 + S),
            Vetoed::Shown
        );
        assert!(store.vetoed(A, one_more, t0 + S));
        assert_eq!(gateways.kept(A), KINDS_MAX + 1);
        drop(seat);
    }

    /// A refusal at a true ceiling asks the store of the note's rows once
    /// a membership epoch. The table alone writes a gateway's vetoes and
    /// a row leaves by a membership write alone: a veto the store lost by
    /// another hand is not found under the same epoch, and the next
    /// membership write has the store asked again.
    #[test]
    fn a_refusal_at_a_true_ceiling_asks_the_store_once_a_membership_epoch() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let workers = filled(&store, &seat, t0);
        let one_more = worker(&store, "http://one-more:8000");
        // Every row stands: the frame is refused, and the store was asked.
        assert_eq!(
            seat.veto(one_more, BREAKER, Some(t0 + 60 * S), t0 + S),
            Vetoed::Over
        );
        // The store loses one worker's veto behind the note, by no
        // membership write: under the same epoch the store is not asked
        // again, and the frame is refused as before.
        HealthSink::gateway_veto(&*store, A, workers[1], None);
        assert!(!store.vetoed(A, workers[1], t0 + S));
        assert_eq!(
            seat.veto(one_more, BREAKER, Some(t0 + 60 * S), t0 + S),
            Vetoed::Over,
            "the store was asked twice under one membership epoch"
        );
        // A membership write moves the epoch: the next frame asks the
        // store, that worker's ends go and the veto is shown.
        let _late = worker(&store, "http://late:8000");
        assert_eq!(
            seat.veto(one_more, BREAKER, Some(t0 + 60 * S), t0 + S),
            Vetoed::Shown
        );
        assert!(store.vetoed(A, one_more, t0 + S));
        assert_eq!(gateways.kept(A), VETOES_MAX - KINDS_MAX + 1);
        drop(seat);
    }

    /// A noted worker removed after a refusal kept the epoch: the store's
    /// removal is a membership write, so the next frame at the ceiling
    /// asks the store again, the removed worker's ends go and the veto is
    /// shown.
    #[test]
    fn a_noted_worker_removed_under_a_kept_epoch_has_the_store_asked_again() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let workers = filled(&store, &seat, t0);
        let one_more = worker(&store, "http://one-more:8000");
        let frame = || seat.veto(one_more, BREAKER, Some(t0 + 60 * S), t0 + S);
        // Refused twice, the second time under the epoch the first kept.
        assert_eq!((frame(), frame()), (Vetoed::Over, Vetoed::Over));
        let revision = WorkersStore::get(&*store, workers[1])
            .expect("the row")
            .revision;
        WorkersStore::remove(&*store, workers[1], revision).expect("removed");
        assert_eq!(frame(), Vetoed::Shown);
        assert!(store.vetoed(A, one_more, t0 + S));
        assert_eq!(gateways.kept(A), VETOES_MAX - KINDS_MAX + 1);
        drop(seat);
    }

    /// A `Hello`'s list replaces the note, and the epoch kept for the old
    /// note's walk goes with it: the first frame at the ceiling after the
    /// list asks the store, under the same membership epoch.
    #[test]
    fn a_list_that_replaces_the_note_drops_the_epoch_its_walk_kept() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let workers = filled(&store, &seat, t0);
        let one_more = worker(&store, "http://one-more:8000");
        let frame = || seat.veto(one_more, BREAKER, Some(t0 + 60 * S), t0 + S);
        // The refusal walks the note and keeps the epoch.
        assert_eq!(frame(), Vetoed::Over);
        // The gateway's list is laid again, whole, with no membership
        // write between.
        assert_eq!(seat.restore(&full_list(&workers, t0), t0), VETOES_MAX);
        // The store loses one worker's veto behind the new note, by no
        // membership write: the frame asks the store all the same, that
        // worker's ends go and the veto is shown.
        HealthSink::gateway_veto(&*store, A, workers[1], None);
        assert_eq!(frame(), Vetoed::Shown);
        assert_eq!(gateways.kept(A), VETOES_MAX - KINDS_MAX + 1);
        drop(seat);
    }

    /// A frame's end nearer than every end the note holds is the note's
    /// earliest from then on, and a walk keeps the earliest end left: each
    /// such end frees its place at the ceiling once past, with no pass of
    /// the sweep between, and the ceiling stands until then.
    #[test]
    fn an_end_a_frame_sets_nearest_frees_its_place_at_the_ceiling_once_past() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let workers = filled(&store, &seat, t0);
        // Two frames move one kind's end each, to five and to seven
        // seconds: the list's ends stand at sixty.
        assert_eq!(
            seat.veto(workers[0], 1, Some(t0 + 5 * S), t0),
            Vetoed::Shown
        );
        assert_eq!(
            seat.veto(workers[1], 1, Some(t0 + 7 * S), t0),
            Vetoed::Shown
        );
        assert_eq!(gateways.kept(A), VETOES_MAX);
        let first = worker(&store, "http://one-more:8000");
        let second = worker(&store, "http://two-more:8000");
        let frame = |w, at| seat.veto(w, BREAKER, Some(t0 + 60 * S), t0 + at);
        // Before the nearest end the ceiling stands; at it one place is
        // free.
        assert_eq!(frame(first, S), Vetoed::Over);
        assert_eq!(frame(first, 5 * S), Vetoed::Shown);
        // The walk kept the earliest end left: the ceiling stands again
        // until that one passes.
        assert_eq!(frame(second, 6 * S), Vetoed::Over);
        assert_eq!(frame(second, 7 * S), Vetoed::Shown);
        assert_eq!(gateways.kept(A), VETOES_MAX);
        drop(seat);
    }

    /// The store is shown the latest end that stands: a kind set with an
    /// earlier end under a later end of another kind leaves the later one
    /// shown.
    #[test]
    fn a_kinds_earlier_end_under_a_later_one_leaves_the_later_end_shown() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let w = worker(&store, "http://a:8000");
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        assert_eq!(seat.veto(w, BREAKER, Some(t0 + 60 * S), t0), Vetoed::Shown);
        assert_eq!(seat.veto(w, LIVENESS, Some(t0 + 5 * S), t0), Vetoed::Shown);
        // Past the earlier end the later one still stands in the store.
        assert!(store.vetoed(A, w, t0 + 30 * S));
        assert!(!store.vetoed(A, w, t0 + 60 * S));
        drop(seat);
    }

    /// A third kind's frame leaves the latest of the other kinds' ends
    /// shown, not the earliest of them.
    #[test]
    fn a_third_kinds_frame_leaves_the_latest_of_the_other_ends_shown() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let w = worker(&store, "http://a:8000");
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let third = BREAKER.max(LIVENESS) + 1;
        assert_eq!(seat.veto(w, BREAKER, Some(t0 + 60 * S), t0), Vetoed::Shown);
        assert_eq!(seat.veto(w, LIVENESS, Some(t0 + 5 * S), t0), Vetoed::Shown);
        // The third kind's own end is the earliest of the three.
        assert_eq!(seat.veto(w, third, Some(t0 + S), t0), Vetoed::Shown);
        assert!(store.vetoed(A, w, t0 + 30 * S));
        assert!(!store.vetoed(A, w, t0 + 60 * S));
        drop(seat);
    }

    /// At a frame, with no pass of the sweep between, an end past its
    /// instant frees its place among a worker's kinds: one kind more is
    /// taken where the ceiling stood.
    #[test]
    fn an_end_past_its_instant_frees_its_place_at_a_frame() {
        let (_elector, store, gateways, t0) = leading(GRACE);
        let w = worker(&store, "http://a:8000");
        let seat = welcomed(&gateways, A, "gateway-a", TERM, t0);
        let kinds = i32::try_from(KINDS_MAX).expect("a small number");
        // The first kind ends a second on, the others a minute on.
        assert_eq!(seat.veto(w, 1, Some(t0 + S), t0), Vetoed::Shown);
        for kind in 2..=kinds {
            assert_eq!(seat.veto(w, kind, Some(t0 + 60 * S), t0), Vetoed::Shown);
        }
        assert_eq!(seat.veto(w, kinds + 1, Some(t0 + 60 * S), t0), Vetoed::Full);
        // At the first kind's instant its place is free.
        assert_eq!(
            seat.veto(w, kinds + 1, Some(t0 + 60 * S), t0 + S),
            Vetoed::Shown,
            "an end past its instant held its place"
        );
        assert_eq!(gateways.kept(A), KINDS_MAX);
        drop(seat);
    }

    #[test]
    fn a_list_holds_a_workers_ceiling_of_kinds_and_no_more() {
        let kinds = i32::try_from(KINDS_MAX).expect("a small number");
        let w = WorkerId(1);
        let list = |last: i32| (1..=last).map(move |kind| (w, kind));
        assert!(!over_kinds(list(kinds)));
        assert!(over_kinds(list(kinds + 1)));
        // A kind listed twice is one kind, and each worker has its count.
        assert!(!over_kinds(list(kinds).chain(list(kinds))));
        let other = (1..=kinds).map(|kind| (WorkerId(2), kind));
        assert!(!over_kinds(list(kinds).chain(other)));
    }
}
