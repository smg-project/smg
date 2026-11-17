//! The log: the last replicated writes of one store, each at its position
//! and under its term. The watched kinds and the load rows each keep
//! their last writes in a ring of their own, in one order of positions.

use std::{
    collections::{vec_deque, VecDeque},
    fmt,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use parking_lot::Mutex;

use crate::{
    store::{Journal, Kinds, LoadSample, MembershipChange, Position, WorkerRow},
    view::WorkerId,
};

/// One replicated write, as the store told it.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// A membership or status write.
    Member {
        /// The membership epoch the write made.
        epoch: u64,
        change: MembershipChange,
        /// The row the change is about, as the write left it in the table;
        /// `None` for a removal.
        row: Option<Arc<WorkerRow>>,
        /// The row the write took out of the table: a removal's row, a
        /// replacement's old incarnation.
        left: Option<Arc<WorkerRow>>,
    },
    /// One rank's load row.
    Load {
        id: WorkerId,
        rank: u16,
        sample: LoadSample,
    },
    /// The store's term rose: the first position of a term, one of the
    /// watched kinds. It changes nothing a standby holds; it is why the
    /// head of a store that leads is never under term `0`, and how a
    /// follower of the log learns the term before the leader's next write.
    Term,
}

/// An event at its place in the history.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub at: Position,
    pub event: Event,
}

/// A reader's place is no longer the log's, or never was: it starts again
/// from a snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lost;

/// One ring: the retained events of its kinds, oldest first.
struct Kept {
    events: VecDeque<Entry>,
    /// The place before the oldest retained event, with its term: the last
    /// event the ring let go, the place of a restart, or the place before
    /// every write.
    floor: Position,
}

impl Kept {
    fn of(capacity: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(capacity),
            floor: Position::default(),
        }
    }

    /// Whether the ring has `at` under its term: a retained event, or the
    /// place before the oldest.
    fn has(&self, at: Position) -> bool {
        at == self.floor
            || self
                .events
                .binary_search_by_key(&at.seq, |entry| entry.at.seq)
                .ok()
                .and_then(|index| self.events.get(index))
                .is_some_and(|entry| entry.at.term == at.term)
    }

    /// The retained events after position `seq`, oldest first.
    fn after(&self, seq: u64) -> vec_deque::Iter<'_, Entry> {
        let from = self.events.partition_point(|entry| entry.at.seq <= seq);
        self.events.range(from..)
    }

    /// The last event's place; the floor while none is retained.
    fn head(&self) -> Position {
        self.events.back().map_or(self.floor, |entry| entry.at)
    }

    /// Appends, letting the oldest go at `capacity`.
    fn push(&mut self, capacity: usize, entry: Entry) {
        if self.events.len() >= capacity {
            // The event the ring lets go is the place before its oldest.
            if let Some(gone) = self.events.pop_front() {
                self.floor = gone.at;
            }
        }
        self.events.push_back(entry);
    }

    fn restart(&mut self, at: Position) {
        self.events.clear();
        self.floor = at;
    }
}

struct Ring {
    /// The watched kinds: membership and status writes, starts of terms.
    watched: Kept,
    /// The load rows.
    loads: Kept,
    /// The last write of any kind, with its term.
    head: Position,
    /// The leader's place for the next write, while a standby applies the
    /// leader's event.
    adopting: Option<Position>,
    /// Whether the store leads the log's term: from its take until it
    /// follows another history.
    leads: bool,
    /// The highest number the log took under no term. No term names the
    /// store's own line of writes, and a restart may set the head below a
    /// number that line took: a place under no term is numbered past it
    /// ([`Ring::fresh`]), so it names one write for as long as the log
    /// lives.
    unnamed: u64,
}

impl Ring {
    /// Whether a reader of `kinds` continues from `at`: the log has that
    /// place under that term (a place is a write of any kind, so either
    /// ring may have it), and every event of `kinds` after it is retained.
    fn continues(&self, kinds: Kinds, at: Position) -> bool {
        (self.watched.has(at) || self.loads.has(at))
            && at.seq >= self.watched.floor.seq
            && (kinds == Kinds::Watched || at.seq >= self.loads.floor.seq)
    }

    /// The number of a place under no term that would stand at `seq`:
    /// `seq`, or the first past every number such a place took before.
    fn fresh(&mut self, seq: u64) -> u64 {
        self.unnamed = seq.max(self.unnamed + 1);
        self.unnamed
    }
}

/// The log of one store: its last replicated writes in one order of
/// positions, in two rings. The watched kinds (membership and status
/// writes, the starts of terms) keep their last writes apart from the load
/// rows, so the load rows' churn lets no membership event go: a reader of
/// the watched kinds keeps its place for as long as that ring holds what
/// follows it.
///
/// A position stands under the term of the leader that wrote it, and a term
/// is taken once, by one run of one store: the log leans on that. The
/// store's own writes are under the log's term while the store leads it,
/// from its take until it follows another history (an applied event, a
/// restart at a snapshot); before a take and after, they stand under no
/// term. Such a place is the store's own: no other store's history
/// continues from it ([`Log::holds`]), and its number is past every
/// number the store wrote under no term before, so a restart that sets
/// the head below an own write gives that write's place to no other.
///
/// The store writes through [`Journal`], under its own locks; a write is one
/// push under the log's lock. A reader copies a batch of at most
/// [`Log::BATCH`] entries out under the same lock and holds it across
/// nothing else. A write of the membership
/// kinds is told to one listener ([`Log::on_watched`]); a load row wakes
/// nobody.
///
/// The bound in bytes: a slot is at most 120 bytes whatever the event. A
/// load row is its slot alone: 65,536 of them are 7.9 MB, held from the
/// start. A membership event's rows stand apart from its slot, 312 bytes a
/// row and its strings: about 0.5 KB an event at a row of a short URL, one
/// model and no label, a second row for a replacement, so 4,096 membership
/// events are about 2 MB at that row.
pub struct Log {
    ring: Mutex<Ring>,
    /// The highest term the log has seen: the store's takes, the events it
    /// applied, the place it restarted at.
    term: AtomicU64,
    /// How many events of the watched kinds are retained.
    watched: usize,
    /// How many load rows are retained; `0` for a log that carries none.
    loads: usize,
    /// Told the place of every write of the watched kinds.
    told: Option<Box<dyn Fn(Position) + Send + Sync>>,
}

impl fmt::Debug for Log {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Log")
            .field("head", &self.head(Kinds::All))
            .field("floor", &self.floor(Kinds::All))
            .field("watched", &self.watched)
            .field("loads", &self.loads)
            .finish_non_exhaustive()
    }
}

impl Log {
    /// The most entries one read copies out, whatever its reader asks: the
    /// longest a reader holds the log's lock.
    pub const BATCH: usize = 256;

    /// A log that retains the last `watched` writes of the watched kinds
    /// (at least one) and the last `loads` load rows. With `loads` of `0`
    /// it is a log of the watched kinds alone: it carries no load row,
    /// and a store that journals to it pays nothing at a load write.
    pub fn new(watched: usize, loads: usize) -> Self {
        let watched = watched.max(1);
        Self {
            ring: Mutex::new(Ring {
                watched: Kept::of(watched),
                loads: Kept::of(loads),
                head: Position::default(),
                adopting: None,
                leads: false,
                unnamed: 0,
            }),
            term: AtomicU64::new(0),
            watched,
            loads,
            told: None,
        }
    }

    /// Calls `told` with the place of every write of the watched kinds, in
    /// the log's order, and with the place of a restart: how a reader of
    /// those kinds learns the log moved without polling it. `told` runs
    /// under the log's lock, inside the store's write: it returns at once
    /// (a watch's send: the last place replaces the one before and nothing
    /// queues), never blocks, and calls nothing of the log or of the
    /// store. A load row is told to nobody.
    #[must_use]
    pub fn on_watched(mut self, told: impl Fn(Position) + Send + Sync + 'static) -> Self {
        self.told = Some(Box::new(told));
        self
    }

    /// The last write of `kinds`, with its term; the default before any.
    /// For the watched kinds it is where a cut of the rows stands: the
    /// rows are as that write left them, whatever load rows came since.
    pub fn head(&self, kinds: Kinds) -> Position {
        let ring = self.ring.lock();
        match kinds {
            Kinds::Watched => ring.watched.head(),
            Kinds::All => ring.head,
        }
    }

    /// The earliest place a reader of `kinds` continues from: every event
    /// of those kinds after it is retained.
    pub fn floor(&self, kinds: Kinds) -> Position {
        let ring = self.ring.lock();
        let (watched, loads) = (ring.watched.floor, ring.loads.floor);
        if kinds == Kinds::All && loads.seq > watched.seq {
            loads
        } else {
            watched
        }
    }

    /// The highest term the log has seen: the store's last take, or a later
    /// term it applied an event of or restarted at. The store's own writes
    /// are under it while the store leads, and under no term otherwise.
    pub fn term(&self) -> u64 {
        self.term.load(Ordering::Relaxed)
    }

    /// Whether a reader of `kinds` that another store's history left at
    /// `at` continues from it: the log has that position under that term,
    /// so the two share the history before it (the replication contract's
    /// rejoin rule), and every event of `kinds` after it is retained. The
    /// log has a place while either ring retains its event, and as the
    /// place before a ring's oldest event, which is the place before every
    /// write until that ring lets its first go. Refused: another term at
    /// the position, a position past the head or one the log let go, and a
    /// position that names no term (nobody led term `0`: two stores may
    /// each hold a write of their own there).
    pub fn holds(&self, kinds: Kinds, at: Position) -> bool {
        (at.term != 0 || at.seq == 0) && self.ring.lock().continues(kinds, at)
    }

    /// The writes of `kinds` after `after`, oldest first, into `out`
    /// (cleared first): at most `at_most` and never more than
    /// [`Log::BATCH`], and all under one term, so a batch splits where the
    /// term changed. Nothing when none followed.
    ///
    /// `after` is where the reader stands: the place [`Log::holds`]
    /// accepted or a cut's, then the `at` of the last entry it read.
    /// [`Lost`] unless the log has that position under that term and
    /// retains every event of `kinds` after it: whatever befell the log
    /// between two reads (a ring let the place go, the log restarted at a
    /// snapshot), a reader is served the history it stands in or nothing.
    pub fn read(
        &self,
        kinds: Kinds,
        after: Position,
        at_most: usize,
        out: &mut Vec<Entry>,
    ) -> Result<(), Lost> {
        out.clear();
        let at_most = at_most.min(Self::BATCH);
        let ring = self.ring.lock();
        if !ring.continues(kinds, after) {
            return Err(Lost);
        }
        let mut watched = ring.watched.after(after.seq).peekable();
        // No position is after the last: a reader of the watched kinds
        // reads no load row.
        let mut loads = ring
            .loads
            .after(match kinds {
                Kinds::All => after.seq,
                Kinds::Watched => u64::MAX,
            })
            .peekable();
        let mut batch_term = None;
        while out.len() < at_most {
            // The two rings in the log's order: the lower position first.
            let next = match (watched.peek(), loads.peek()) {
                (Some(event), Some(load)) if event.at.seq < load.at.seq => watched.next(),
                (Some(_), None) => watched.next(),
                (_, Some(_)) => loads.next(),
                (None, None) => None,
            };
            let Some(entry) = next else {
                break;
            };
            if batch_term.is_some_and(|of_batch| of_batch != entry.at.term) {
                break;
            }
            batch_term = Some(entry.at.term);
            out.push(entry.clone());
        }
        Ok(())
    }

    fn push(&self, event: Event) {
        let mut ring = self.ring.lock();
        let at = match ring.adopting.take() {
            Some(at) => {
                self.follow(&mut ring, at);
                at
            }
            // The term is read and written under the log's lock: the terms
            // of two positions a leader wrote are in the positions' order.
            None => {
                let term = self.own(&ring);
                let mut seq = ring.head.seq + 1;
                if term == 0 {
                    // Its number is the write's whole name: past the head,
                    // and past every own write a restart left above it.
                    seq = ring.fresh(seq);
                }
                Position { seq, term }
            }
        };
        self.append(&mut ring, at, event);
    }

    /// The term of a write of the store's own: the log's while the store
    /// leads it, none otherwise. A store that follows another history, or
    /// never led, writes under no term: after a snapshot's place the term
    /// it led already named other writes.
    fn own(&self, ring: &Ring) -> u64 {
        if ring.leads {
            self.term.load(Ordering::Relaxed)
        } else {
            0
        }
    }

    /// The store follows the history `at` stands in, a leader's event that
    /// landed or a snapshot's place: the term it led is over, and no later
    /// take is at or below the term it followed.
    fn follow(&self, ring: &mut Ring, at: Position) {
        ring.leads = false;
        self.term.fetch_max(at.term, Ordering::Relaxed);
    }

    fn append(&self, ring: &mut Ring, at: Position, event: Event) {
        ring.head = at;
        let entry = Entry { at, event };
        if matches!(entry.event, Event::Load { .. }) {
            ring.loads.push(self.loads, entry);
        } else {
            ring.watched.push(self.watched, entry);
            self.tell(at);
        }
    }

    fn tell(&self, at: Position) {
        if let Some(told) = &self.told {
            told(at);
        }
    }

    /// A standby's application of the leader's event at `at`: while the
    /// returned guard stands, the store's next write is journaled at that
    /// position under that term, the leader's own, so a position and its
    /// term name one history on every replica that holds them. Refused
    /// unless `at` is the position after the head, under a named term not
    /// below any the log has seen, and the head is a leader's write or the
    /// start: a gap, a replay, a falling term and a write of the store's
    /// own under no term are not where a leader's history continues. Nor
    /// is an event under the term this store leads: taken once, that term
    /// has one writer.
    ///
    /// A store that follows has one writer of journaled writes, the one
    /// that applies the leader's batches. Once an event lands, the term the
    /// store led is over: a write of its own stands under no term until
    /// the store takes a term again.
    pub fn adopt(&self, at: Position) -> Result<Adopting<'_>, Lost> {
        let mut ring = self.ring.lock();
        let unled = ring.head.term == 0 && ring.head.seq != 0;
        let term = self.term.load(Ordering::Relaxed);
        let own = ring.leads && at.term == term;
        if unled || own || at.term == 0 || at.seq != ring.head.seq + 1 || at.term < term {
            return Err(Lost);
        }
        ring.adopting = Some(at);
        Ok(Adopting { log: self, at })
    }

    /// Start again at `at`, the position of a snapshot this store was
    /// restored from: nothing is retained, the head of every kind is `at`,
    /// and the next write follows it. A reader that stood anywhere else is
    /// [`Lost`] at its next read, and the listener is told. The store
    /// follows the snapshot's leader from here: the term it led is over,
    /// so a late write of its own takes no position that term wrote before
    /// the restart, and none a write of its own took under no term: it is
    /// numbered past both.
    ///
    /// A snapshot's place under no term is another store's number. Where
    /// this store's own line reached it, the log stands at the next number
    /// of that line instead, and that is the head: a reader of the own
    /// write that stood there is [`Lost`].
    pub fn restart(&self, mut at: Position) {
        let mut ring = self.ring.lock();
        if at.term == 0 && at.seq != 0 {
            at.seq = ring.fresh(at.seq);
        }
        ring.watched.restart(at);
        ring.loads.restart(at);
        ring.head = at;
        ring.adopting = None;
        self.follow(&mut ring, at);
        self.tell(at);
    }
}

/// One leader's event being applied to a standby's store ([`Log::adopt`]).
#[derive(Debug)]
pub struct Adopting<'a> {
    log: &'a Log,
    at: Position,
}

impl Adopting<'_> {
    /// Whether the store wrote: the head is the adopted position. One of a
    /// leader's events may write nothing to its twin, a term's start the
    /// store was told before, and the standby marks its place
    /// ([`Adopting::mark`]). A store that wrote nothing for any other event
    /// is not the leader's twin: a worker's load rows stand between its
    /// registration and its leaving in the leader's log, so each lands.
    pub fn landed(&self) -> bool {
        self.log.head(Kinds::All) == self.at
    }

    /// Applies a leader's event that writes nothing to this store, the
    /// first position of a term: the place alone is journaled, and the next
    /// event follows it.
    pub fn mark(&self) {
        let mut ring = self.log.ring.lock();
        if let Some(at) = ring.adopting.take() {
            self.log.follow(&mut ring, at);
            self.log.append(&mut ring, at, Event::Term);
        }
    }
}

impl Drop for Adopting<'_> {
    fn drop(&mut self) {
        // A place no write took is not kept for a later one.
        self.log.ring.lock().adopting = None;
    }
}

impl Journal for Log {
    fn member(
        &self,
        epoch: u64,
        change: MembershipChange,
        row: Option<&WorkerRow>,
        left: Option<&WorkerRow>,
    ) {
        // The rows are copied before the log's lock is taken.
        let copy = |row: &WorkerRow| Arc::new(row.clone());
        self.push(Event::Member {
            epoch,
            change,
            row: row.map(copy),
            left: left.map(copy),
        });
    }

    fn load(&self, id: WorkerId, rank: u16, sample: &LoadSample) {
        // A log of the watched kinds alone: the row takes no position.
        if self.loads == 0 {
            return;
        }
        self.push(Event::Load {
            id,
            rank,
            sample: *sample,
        });
    }

    fn term_rose(&self, term: u64) {
        let mut ring = self.ring.lock();
        // Told while that term's first position is being applied, the term
        // is its leader's: the place is journaled and the store follows.
        if let Some(at) = ring.adopting.filter(|at| at.term == term) {
            ring.adopting = None;
            self.follow(&mut ring, at);
            self.append(&mut ring, at, Event::Term);
            return;
        }
        if term <= self.term.load(Ordering::Relaxed) {
            return;
        }
        self.term.store(term, Ordering::Relaxed);
        // The store took a lease: it applies no more of another leader's
        // events, and the term starts with a position of its own.
        ring.adopting = None;
        ring.leads = true;
        let at = Position {
            seq: ring.head.seq + 1,
            term,
        };
        self.append(&mut ring, at, Event::Term);
    }

    fn carries_loads(&self) -> bool {
        self.loads != 0
    }

    fn head(&self, kinds: Kinds) -> Position {
        Log::head(self, kinds)
    }
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Instant};

    use super::*;
    use crate::{
        replication::tests::ascends,
        store::{
            Kinds::{All, Watched},
            LoadSource,
        },
        view::LoadRow,
    };

    /// The mark a term's first position reads as.
    const TERM: u32 = u32::MAX;
    /// The mark a membership event reads as, above its epoch.
    const MEMBER: u32 = 1_000_000;

    fn load(log: &Log, running: u32) {
        let at = Instant::now();
        let sample = LoadSample {
            row: LoadRow {
                running,
                ..LoadRow::default()
            },
            received_at: at,
            sampled_at: at,
            source: LoadSource::Stream,
        };
        log.load(WorkerId(7), 0, &sample);
    }

    /// A write of the watched kinds, marked by its epoch.
    fn member(log: &Log, epoch: u32) {
        let change = MembershipChange::Relabelled {
            id: WorkerId(u64::from(epoch)),
        };
        log.member(u64::from(epoch), change, None, None);
    }

    /// Each event's mark, in order: a load row's `running`, a membership
    /// event's epoch above [`MEMBER`], [`TERM`] for a term's start.
    fn marks(entries: &[Entry]) -> Vec<u32> {
        entries
            .iter()
            .map(|entry| match &entry.event {
                Event::Load { sample, .. } => sample.row.running,
                Event::Member { epoch, .. } => {
                    MEMBER + u32::try_from(*epoch).expect("a test's epoch")
                }
                Event::Term => TERM,
            })
            .collect()
    }

    fn places(entries: &[Entry]) -> Vec<Position> {
        entries.iter().map(|entry| entry.at).collect()
    }

    fn at(seq: u64, term: u64) -> Position {
        Position { seq, term }
    }

    /// Every entry of `log`, read from its start.
    fn all(log: &Log) -> Vec<Entry> {
        all_after(log, All, Position::default())
    }

    /// Every entry of `kinds` after `after`, each batch read from the place
    /// of the last entry before it. The walk fails where a batch does not
    /// ascend from that place or where it passes what the two rings hold:
    /// a read that handed the place back would append without end.
    fn all_after(log: &Log, kinds: Kinds, after: Position) -> Vec<Entry> {
        let mut entries: Vec<Entry> = Vec::new();
        let mut out = Vec::new();
        loop {
            let from = entries.last().map_or(after, |entry| entry.at);
            if log.read(kinds, from, 3, &mut out) != Ok(()) || out.is_empty() {
                return entries;
            }
            ascends(from, &out);
            entries.append(&mut out);
            assert!(
                entries.len() <= log.watched + log.loads,
                "a walk of more than the two rings hold"
            );
        }
    }

    /// Applies a leader's load row at `at`, as a standby does.
    fn apply(log: &Log, at: Position, mark: u32) {
        let adopting = log.adopt(at).expect("the next position");
        load(log, mark);
        assert!(adopting.landed());
    }

    /// Applies a leader's entries in order, as a standby does: each by the
    /// store's own write under the guard, a term's start by the mark.
    fn applies(follower: &Log, led: &[Entry]) {
        for entry in led {
            let adopting = follower.adopt(entry.at).expect("the next position");
            assert!(!adopting.landed());
            match &entry.event {
                Event::Load { id, rank, sample } => follower.load(*id, *rank, sample),
                Event::Term => adopting.mark(),
                Event::Member {
                    epoch,
                    change,
                    row,
                    left,
                } => follower.member(*epoch, *change, row.as_deref(), left.as_deref()),
            }
            assert!(adopting.landed());
        }
    }

    #[test]
    fn a_slot_is_at_most_120_bytes_and_a_membership_row_stands_apart() {
        // The log's bound in bytes is told from these two: a ring's slots,
        // and apart from its slot a membership event's rows (two for a
        // replacement), each with its strings.
        assert!(size_of::<Entry>() <= 120);
        assert!(size_of::<WorkerRow>() <= 312);
    }

    #[test]
    fn one_count_of_positions_runs_through_both_rings_and_a_term_starts_with_its_own() {
        let log = Log::new(8, 8);
        assert_eq!(log.head(All), Position::default());
        assert_eq!(log.floor(All), Position::default());
        load(&log, 10);
        log.term_rose(3);
        assert_eq!(
            log.head(All),
            at(2, 3),
            "the take is the term's first position"
        );
        load(&log, 11);
        member(&log, 1);
        load(&log, 12);
        assert_eq!(log.head(All), at(5, 3));
        assert_eq!(
            log.head(Watched),
            at(4, 3),
            "the last write of the watched kinds"
        );
        let mut out = Vec::new();
        assert_eq!(log.read(All, at(0, 0), 8, &mut out), Ok(()));
        assert_eq!(
            places(&out),
            [at(1, 0)],
            "a batch stops where the term changes"
        );
        assert_eq!(log.read(All, at(1, 0), 8, &mut out), Ok(()));
        assert_eq!(places(&out), [at(2, 3), at(3, 3), at(4, 3), at(5, 3)]);
        assert_eq!(
            marks(&out),
            [TERM, 11, MEMBER + 1, 12],
            "the two rings in the log's order"
        );
        assert_eq!(log.read(Watched, at(0, 0), 8, &mut out), Ok(()));
        assert_eq!(
            places(&out),
            [at(2, 3), at(4, 3)],
            "a reader of the watched kinds reads no load row"
        );
    }

    #[test]
    fn a_term_never_falls_and_only_a_rise_takes_a_position() {
        let log = Log::new(8, 8);
        log.term_rose(5);
        log.term_rose(2);
        log.term_rose(5);
        assert_eq!(log.term(), 5);
        assert_eq!(log.head(All), at(1, 5), "one rise, one position");
        load(&log, 1);
        assert_eq!(log.head(All), at(2, 5));
    }

    #[test]
    fn the_load_rows_churn_lets_no_membership_event_go() {
        let log = Log::new(4, 4);
        log.term_rose(1);
        member(&log, 1);
        // Twenty load rows at positions 3 to 22: their ring keeps the last
        // four.
        for mark in 0..20 {
            load(&log, mark);
        }
        member(&log, 2);
        assert_eq!(
            log.floor(All),
            at(18, 1),
            "the last load row its ring let go"
        );
        assert_eq!(
            log.floor(Watched),
            at(0, 0),
            "no membership event left its ring"
        );
        let mut out = Vec::new();
        // The watcher that last read position 2 reads the next membership
        // event, twenty load rows later.
        assert!(log.holds(Watched, at(2, 1)));
        assert_eq!(log.read(Watched, at(2, 1), 8, &mut out), Ok(()));
        assert_eq!(places(&out), [at(23, 1)]);
        // A reader of every kind that stood there lost the load rows
        // between.
        assert!(!log.holds(All, at(2, 1)));
        assert_eq!(log.read(All, at(2, 1), 8, &mut out), Err(Lost));
        // The place before the oldest load row is the earliest a reader of
        // every kind continues from.
        assert_eq!(log.read(All, at(18, 1), 8, &mut out), Ok(()));
        assert_eq!(marks(&out), [16, 17, 18, 19, MEMBER + 2]);
        // A place is a write of any kind: a watcher seeded from the log's
        // head, a load row's place, is served while the log has that row.
        assert_eq!(log.read(Watched, at(20, 1), 8, &mut out), Ok(()));
        assert_eq!(places(&out), [at(23, 1)]);
        // Once the log let the row go it is told so, though the watched
        // kinds' ring is whole after it: no term is taken on trust.
        assert_eq!(log.read(Watched, at(17, 1), 8, &mut out), Err(Lost));
        assert_eq!(
            log.read(Watched, at(18, 7), 8, &mut out),
            Err(Lost),
            "the load ring's floor under another term"
        );
    }

    #[test]
    fn a_place_the_load_ring_let_go_is_never_served_by_a_term_read_off_its_neighbours() {
        let log = Log::new(4, 4);
        log.term_rose(1);
        // Eight load rows at positions 2 to 9: their ring keeps the last
        // four.
        for mark in 0..8 {
            load(&log, mark);
        }
        log.term_rose(2);
        let mut out = Vec::new();
        // Position 3 was a load row under term 1, between two starts of
        // terms the log still has. The log let the row go and says so,
        // whichever of the two terms the reader names.
        for term in [1, 2] {
            assert!(!log.holds(Watched, at(3, term)), "{term}");
            assert_eq!(
                log.read(Watched, at(3, term), 8, &mut out),
                Err(Lost),
                "{term}"
            );
        }
        // A reader at the start of the first term reads the second's.
        assert_eq!(log.read(Watched, at(1, 1), 8, &mut out), Ok(()));
        assert_eq!(places(&out), [at(10, 2)]);
    }

    #[test]
    fn a_reader_continues_only_from_a_place_the_log_has_under_its_term() {
        // A store that wrote before any lease: the position stands under
        // term 0, which names no term. A standby does not continue from it;
        // the log's own reader, which read it here, does.
        let unled = Log::new(4, 4);
        load(&unled, 1);
        assert_eq!(unled.head(All), at(1, 0));
        assert!(!unled.holds(All, at(1, 0)), "had, but under no term");
        assert!(unled.holds(All, at(0, 0)));
        let mut out = Vec::new();
        assert_eq!(unled.read(All, at(1, 0), 4, &mut out), Ok(()));

        let log = Log::new(4, 4);
        assert!(
            log.holds(All, at(0, 0)),
            "an empty log is joined at its start"
        );
        assert!(!log.holds(All, at(1, 1)), "past the head of an empty log");
        log.term_rose(2);
        load(&log, 1);
        load(&log, 2);
        log.term_rose(3);
        assert_eq!(places(&all(&log)), [at(1, 2), at(2, 2), at(3, 2), at(4, 3)]);
        for kinds in [All, Watched] {
            assert!(log.holds(kinds, at(0, 0)), "no first write left its ring");
            assert!(log.holds(kinds, at(3, 2)), "a load row's place");
            assert!(log.holds(kinds, at(4, 3)), "a term's first place");
            assert!(!log.holds(kinds, at(3, 3)), "had under another term");
            assert!(!log.holds(kinds, at(4, 2)), "had under another term");
            assert!(!log.holds(kinds, at(5, 3)), "past the head");
            assert!(!log.holds(kinds, at(3, 0)), "a position that names no term");
            assert!(!log.holds(kinds, at(0, 2)), "a term with no position");
        }
        // The watched kinds' ring of four takes three more and lets its first
        // write go: that write is the place before the ring's oldest, under
        // its own term.
        for epoch in 1..=3 {
            member(&log, epoch);
        }
        assert_eq!(log.floor(Watched), at(1, 2));
        assert!(
            !log.holds(Watched, at(0, 0)),
            "the first write left its ring"
        );
        assert!(
            log.holds(Watched, at(1, 2)),
            "the place before the oldest retained"
        );
        assert!(
            !log.holds(Watched, at(1, 3)),
            "that place under another term"
        );
        assert!(
            log.holds(All, at(1, 2)),
            "no load row after it left its ring"
        );
        member(&log, 4);
        assert_eq!(log.floor(Watched), at(4, 3));
        assert!(!log.holds(Watched, at(1, 2)), "behind the ring");
        assert!(
            !log.holds(Watched, at(3, 2)),
            "a retained load row's place, a membership event after it gone"
        );
        assert!(log.holds(Watched, at(4, 3)));
    }

    #[test]
    fn a_read_is_bounded_and_a_reader_at_the_head_reads_nothing() {
        let log = Log::new(4, 16);
        for mark in 0..10 {
            load(&log, mark);
        }
        let mut out = Vec::new();
        assert_eq!(log.read(All, at(0, 0), 4, &mut out), Ok(()));
        assert_eq!(marks(&out), [0, 1, 2, 3]);
        assert_eq!(log.read(All, at(4, 0), 100, &mut out), Ok(()));
        assert_eq!(marks(&out), [4, 5, 6, 7, 8, 9]);
        assert_eq!(log.read(All, at(10, 0), 4, &mut out), Ok(()));
        assert!(out.is_empty(), "the reader is at the head");
        assert_eq!(
            log.read(All, at(11, 0), 4, &mut out),
            Err(Lost),
            "past the head"
        );
        assert_eq!(
            log.read(All, at(4, 1), 4, &mut out),
            Err(Lost),
            "a place the log has under another term"
        );
        assert_eq!(log.read(All, at(4, 0), 0, &mut out), Ok(()));
        assert!(out.is_empty(), "a read of nothing says the place stands");
        // Whatever its reader asks, one read copies the log's own bound.
        let log = Log::new(4, 2 * Log::BATCH);
        for mark in 0..300 {
            load(&log, mark);
        }
        assert_eq!(log.read(All, at(0, 0), usize::MAX, &mut out), Ok(()));
        assert_eq!(out.len(), Log::BATCH);
    }

    #[test]
    fn a_ring_keeps_its_last_writes_and_a_reader_behind_it_is_lost() {
        let log = Log::new(4, 4);
        for mark in 0..10 {
            load(&log, mark);
        }
        assert_eq!(log.head(All), at(10, 0));
        assert_eq!(log.floor(All), at(6, 0));
        assert_eq!(log.floor(Watched), at(0, 0));
        let mut out = vec![Entry {
            at: at(99, 99),
            event: Event::Term,
        }];
        assert_eq!(log.read(All, at(5, 0), 8, &mut out), Err(Lost));
        assert!(
            out.is_empty(),
            "a lost read leaves nothing of an earlier one"
        );
        // The reader that applied position 6 reads the oldest retained next.
        assert_eq!(log.read(All, at(6, 0), 8, &mut out), Ok(()));
        assert_eq!(marks(&out), [6, 7, 8, 9]);
        assert_eq!(out.first().map(|entry| entry.at), Some(at(7, 0)));
        // A log asked for no slot of the watched kinds keeps one.
        let log = Log::new(0, 0);
        member(&log, 1);
        member(&log, 2);
        assert_eq!(log.floor(Watched), at(1, 0));
        assert_eq!(marks(&all_after(&log, Watched, at(1, 0))), [MEMBER + 2]);
    }

    #[test]
    fn writers_on_many_threads_leave_dense_positions_under_rising_terms() {
        const WRITERS: u32 = 4;
        const EACH: u32 = 500;
        const TERMS: u32 = EACH / 50;
        let log = Arc::new(Log::new(64, (WRITERS * EACH) as usize));
        thread::scope(|scope| {
            for writer in 0..WRITERS {
                let log = Arc::clone(&log);
                scope.spawn(move || {
                    for write in 0..EACH {
                        if write % 50 == 0 {
                            log.term_rose(u64::from(write / 50 + 1));
                        }
                        if write % 100 == 7 {
                            member(&log, writer * EACH + write);
                        } else {
                            load(&log, writer * EACH + write);
                        }
                    }
                });
            }
        });
        // Every writer asks for the same ten terms: each rises once.
        let total = u64::from(WRITERS * EACH + TERMS);
        assert_eq!(log.head(All).seq, total);
        let entries = all(&log);
        assert_eq!(entries.len() as u64, total);
        for (index, pair) in entries.windows(2).enumerate() {
            assert_eq!(pair[1].at.seq, pair[0].at.seq + 1, "a gap after {index}");
            assert!(
                pair[1].at.term >= pair[0].at.term,
                "a term fell after {index}"
            );
        }
        let mut seen = marks(&entries);
        seen.sort_unstable();
        let mut wanted: Vec<u32> = (0..WRITERS * EACH)
            .map(|write| {
                if write % EACH % 100 == 7 {
                    MEMBER + write
                } else {
                    write
                }
            })
            .collect();
        wanted.extend([TERM; TERMS as usize]);
        wanted.sort_unstable();
        assert_eq!(seen, wanted);
        // A term's first position is its mark.
        for term in 1..=u64::from(TERMS) {
            let first = entries.iter().find(|entry| entry.at.term == term);
            assert_eq!(
                first.map(|entry| &entry.event),
                Some(&Event::Term),
                "{term}"
            );
        }
        // The watched kinds alone, read from the start, are the same
        // events in the same order.
        let members: Vec<Entry> = entries
            .iter()
            .filter(|entry| !matches!(entry.event, Event::Load { .. }))
            .cloned()
            .collect();
        assert_eq!(all_after(&log, Watched, Position::default()), members);
    }

    #[test]
    fn the_log_tells_every_write_of_the_watched_kinds_and_no_load_row() {
        let told = Arc::new(Mutex::new(Vec::new()));
        let log = {
            let told = Arc::clone(&told);
            Log::new(8, 8).on_watched(move |at| told.lock().push(at))
        };
        load(&log, 1);
        member(&log, 1);
        log.term_rose(2);
        load(&log, 2);
        assert_eq!(*told.lock(), [at(2, 0), at(3, 2)]);
        // A standby's log tells what it applies, and where it restarts.
        log.adopt(at(5, 3)).expect("the next position").mark();
        apply(&log, at(6, 3), 3);
        log.restart(at(9, 4));
        assert_eq!(told.lock()[2..], [at(5, 3), at(9, 4)]);
    }

    #[test]
    fn a_log_of_the_watched_kinds_alone_takes_no_load_row() {
        let log = Log::new(4, 0);
        assert!(!log.carries_loads());
        assert!(Log::new(4, 1).carries_loads());
        load(&log, 1);
        assert_eq!(
            log.head(All),
            Position::default(),
            "the row took no position"
        );
        member(&log, 1);
        load(&log, 2);
        member(&log, 2);
        assert_eq!(places(&all(&log)), [at(1, 0), at(2, 0)]);
        assert_eq!(marks(&all(&log)), [MEMBER + 1, MEMBER + 2]);
    }

    #[test]
    fn an_applied_event_takes_the_leaders_position_and_term() {
        let leader = Log::new(16, 16);
        load(&leader, 1);
        leader.term_rose(4);
        load(&leader, 2);
        leader.term_rose(6);
        member(&leader, 1);
        load(&leader, 3);
        let led = all(&leader);
        assert_eq!(
            places(&led),
            [at(1, 0), at(2, 4), at(3, 4), at(4, 6), at(5, 6), at(6, 6)]
        );
        // The follower led a term of its own once: an applied write never
        // takes it. It joined from a snapshot at the leader's first take.
        let follower = Log::new(16, 16);
        follower.term_rose(3);
        follower.restart(at(2, 4));
        applies(&follower, &led[2..]);
        assert_eq!(follower.head(All), leader.head(All));
        assert_eq!(follower.head(Watched), leader.head(Watched));
        assert_eq!(follower.floor(All), at(2, 4));
        assert_eq!(all_after(&follower, All, at(2, 4)), led[2..]);
        for held in [at(3, 4), at(5, 6), at(6, 6)] {
            assert!(follower.holds(All, held), "{held:?}");
            assert!(leader.holds(All, held), "{held:?}");
        }
        assert!(
            !follower.holds(All, at(3, 3)),
            "never under the term the follower led"
        );
        assert_eq!(follower.term(), 6, "the highest term it followed");
    }

    #[test]
    fn an_applied_event_is_refused_at_a_gap_a_replay_or_a_falling_term() {
        let log = Log::new(8, 8);
        assert_eq!(log.adopt(at(2, 3)).err(), Some(Lost), "a gap");
        assert_eq!(log.adopt(at(1, 0)).err(), Some(Lost), "no term");
        apply(&log, at(1, 3), 1);
        assert_eq!(log.head(All), at(1, 3));
        assert_eq!(log.adopt(at(1, 3)).err(), Some(Lost), "a replay");
        assert_eq!(log.adopt(at(3, 3)).err(), Some(Lost), "a gap");
        assert_eq!(log.adopt(at(2, 2)).err(), Some(Lost), "a falling term");
        assert!(log.adopt(at(2, 3)).is_ok());
        // A term below one the log has seen, wherever its head stands.
        let log = Log::new(8, 8);
        log.term_rose(7);
        log.restart(at(10, 6));
        assert_eq!(
            log.adopt(at(11, 6)).err(),
            Some(Lost),
            "below a term the log took"
        );
        assert!(log.adopt(at(11, 7)).is_ok());
    }

    #[test]
    fn an_own_write_under_no_term_takes_no_number_an_earlier_one_took() {
        let log = Log::new(8, 8);
        let mut out = Vec::new();
        // A store that never restarted writes one number after another.
        member(&log, 1);
        assert_eq!(log.head(All), at(1, 0));
        assert_eq!(log.read(All, at(1, 0), 8, &mut out), Ok(()));
        // Restored at a leader's place, it writes once more, and a reader
        // reads that write.
        log.restart(at(2, 2));
        member(&log, 2);
        assert_eq!(log.read(Watched, at(2, 2), 8, &mut out), Ok(()));
        assert_eq!(places(&out), [at(3, 0)]);
        // Restored again at the same number: the log no longer has the
        // write that reader stands at, and the next one is numbered past
        // it, so the reader is never served another line as its own.
        log.restart(at(2, 3));
        assert_eq!(log.read(Watched, at(3, 0), 8, &mut out), Err(Lost));
        member(&log, 3);
        for kinds in [All, Watched] {
            assert_eq!(log.read(kinds, at(3, 0), 8, &mut out), Err(Lost));
        }
        assert_eq!(log.head(All), at(4, 0));
        // The restart's own place, a leader's, continues its reader.
        assert_eq!(log.read(All, at(2, 3), 8, &mut out), Ok(()));
        assert_eq!(places(&out), [at(4, 0)]);
        assert_eq!(marks(&out), [MEMBER + 3]);
    }

    #[test]
    fn a_reader_at_a_restarted_logs_own_write_reads_on() {
        let log = Log::new(8, 8);
        let mut out = Vec::new();
        log.restart(at(2, 2));
        member(&log, 1);
        // The head is the store's own write, where a cut of the store
        // stands: its reader continues from it, with nothing to read yet.
        let cut = log.head(Watched);
        assert_eq!(cut, at(3, 0));
        for kinds in [All, Watched] {
            assert_eq!(log.read(kinds, cut, 8, &mut out), Ok(()));
            assert!(out.is_empty());
        }
        member(&log, 2);
        load(&log, 9);
        assert_eq!(log.read(All, cut, 8, &mut out), Ok(()));
        assert_eq!(places(&out), [at(4, 0), at(5, 0)]);
        assert_eq!(marks(&out), [MEMBER + 2, 9]);
    }

    /// One move of a store's log.
    #[derive(Clone, Copy, Debug)]
    enum Move {
        /// The store is restored from a snapshot at this place.
        Restart(u64, u64),
        /// A membership write of the store's own.
        Own,
        /// A load row of the store's own.
        Row,
        /// A leader's load row, applied at this place.
        Led(u64, u64),
        /// The store takes this term.
        Take(u64),
    }

    fn make(log: &Log, moves: &[Move]) {
        for made in moves {
            match *made {
                Move::Restart(seq, term) => log.restart(at(seq, term)),
                Move::Own => member(log, 1),
                Move::Row => load(log, 1),
                Move::Led(seq, term) => apply(log, at(seq, term), 1),
                Move::Take(term) => log.term_rose(term),
            }
        }
    }

    #[test]
    fn a_place_under_no_term_names_one_write_for_as_long_as_the_log_lives() {
        use Move::{Led, Own, Restart, Row, Take};
        // A reader stands where the first moves leave the head, a place
        // under no term: a write of the store's own, or a restart at
        // another store's place. A restart follows: the reader is lost at
        // every read after it, whatever is written since, and a reader
        // that starts again at the head reads on.
        let walks: [(&str, &[Move], &[Move]); 11] = [
            (
                "one number under another term",
                &[Restart(2, 2), Own],
                &[Restart(2, 3), Own, Own],
            ),
            (
                "the same place twice",
                &[Restart(2, 2), Own],
                &[Restart(2, 2), Own, Own],
            ),
            (
                "an earlier number",
                &[Restart(4, 2), Own],
                &[Restart(2, 3), Own, Own, Own, Own],
            ),
            (
                "a later place",
                &[Restart(2, 2), Own],
                &[Restart(5, 3), Own, Own],
            ),
            (
                "a leader's event first",
                &[Restart(2, 2), Led(3, 2), Own],
                &[Restart(2, 3), Led(3, 3), Own, Own],
            ),
            (
                "a take since",
                &[Restart(2, 2), Own],
                &[Restart(2, 3), Own, Take(4), Own],
            ),
            (
                "a load row",
                &[Restart(2, 2), Row],
                &[Restart(2, 3), Row, Row],
            ),
            (
                "three restarts",
                &[Restart(2, 2), Own],
                &[Restart(2, 3), Restart(2, 4), Own, Own],
            ),
            (
                "writes before the first restart",
                &[Own, Own],
                &[Restart(1, 2), Own, Own],
            ),
            (
                "a restart's own place under no term",
                &[Restart(9, 0)],
                &[Restart(2, 3), Own, Own, Own, Own, Own, Own, Own],
            ),
            (
                "the same place under no term twice",
                &[Restart(9, 0)],
                &[Restart(9, 0), Own],
            ),
        ];
        for (walk, first, then) in walks {
            let log = Log::new(8, 8);
            let mut out = Vec::new();
            make(&log, first);
            let stood = log.head(All);
            assert_eq!(stood.term, 0, "{walk}");
            for (nth, made) in then.iter().enumerate() {
                make(&log, std::slice::from_ref(made));
                for kinds in [All, Watched] {
                    let read = log.read(kinds, stood, 8, &mut out);
                    assert_eq!(read, Err(Lost), "{walk}: {kinds:?} after move {nth}");
                }
                let read = log.read(All, log.head(All), 8, &mut out);
                assert_eq!(read, Ok(()), "{walk}: at the head after move {nth}");
            }
        }
    }

    #[test]
    fn a_restart_at_a_place_under_no_term_stands_past_every_own_write() {
        let log = Log::new(8, 8);
        let mut out = Vec::new();
        for epoch in 1..=5 {
            member(&log, epoch);
        }
        assert_eq!(log.head(All), at(5, 0));
        // Another store's place under no term, at a number this store's
        // own line reached: the reader of the fifth write is not served
        // the other store's history, since the log stands at the next
        // number of its own line.
        log.restart(at(5, 0));
        member(&log, 6);
        assert_eq!(log.read(All, at(5, 0), 8, &mut out), Err(Lost));
        assert_eq!(log.floor(All), at(6, 0));
        assert_eq!(log.read(All, at(6, 0), 8, &mut out), Ok(()));
        assert_eq!(places(&out), [at(7, 0)]);
        // A number no own write reached is the log's place as told.
        log.restart(at(9, 0));
        assert_eq!(log.head(All), at(9, 0));
        member(&log, 7);
        assert_eq!(log.head(All), at(10, 0));
        // The place before every write is no write's number.
        log.restart(at(0, 0));
        assert_eq!(log.head(All), at(0, 0));
        member(&log, 8);
        assert_eq!(log.read(All, at(0, 0), 8, &mut out), Ok(()));
        assert_eq!(places(&out), [at(11, 0)]);
    }

    #[test]
    fn a_leaders_positions_follow_its_head_whatever_its_own_line_reached() {
        let leader = Log::new(8, 8);
        // Own writes under no term reach position 4. A restart sets the
        // head below them, and the store takes a term.
        leader.restart(at(2, 2));
        member(&leader, 1);
        member(&leader, 2);
        assert_eq!(leader.head(All), at(4, 0));
        leader.restart(at(1, 5));
        leader.term_rose(6);
        load(&leader, 1);
        member(&leader, 3);
        // A term's positions are one after another from the head: what a
        // follower applies, and no gap is applied.
        let led = all_after(&leader, All, at(1, 5));
        assert_eq!(places(&led), [at(2, 6), at(3, 6), at(4, 6)]);
        let follower = Log::new(8, 8);
        follower.restart(at(1, 5));
        applies(&follower, &led);
        assert_eq!(follower.head(All), leader.head(All));
        // Position 4 is the leader's now: the reader of the own write that
        // stood there is lost.
        let mut out = Vec::new();
        assert_eq!(leader.read(All, at(4, 0), 8, &mut out), Err(Lost));
    }

    #[test]
    fn a_place_no_write_took_is_not_kept_for_the_stores_next_write() {
        let log = Log::new(8, 8);
        log.term_rose(2);
        let adopting = log.adopt(at(2, 9)).expect("the next position");
        assert!(!adopting.landed());
        drop(adopting);
        load(&log, 1);
        assert_eq!(
            log.head(All),
            at(2, 2),
            "the store's own write, under its own term"
        );
    }

    #[test]
    fn a_take_ends_an_application_and_starts_its_term_after_the_head() {
        let log = Log::new(8, 8);
        apply(&log, at(1, 3), 1);
        let adopting = log.adopt(at(2, 3)).expect("the next position");
        // The store takes a lease while another leader's event is half
        // applied: the event's place is dropped, never taken by the mark.
        log.term_rose(4);
        assert!(!adopting.landed());
        assert_eq!(log.head(All), at(2, 4));
        load(&log, 2);
        assert_eq!(places(&all(&log)), [at(1, 3), at(2, 4), at(3, 4)]);
        assert_eq!(marks(&all(&log)), [1, TERM, 2]);
    }

    #[test]
    fn a_log_restarted_at_a_snapshot_has_that_place_alone_and_continues_after_it() {
        let log = Log::new(8, 8);
        log.term_rose(4);
        for mark in 0..3 {
            load(&log, mark);
        }
        log.restart(at(100, 7));
        assert_eq!(log.term(), 7, "the term of the snapshot's place");
        for kinds in [All, Watched] {
            assert_eq!(log.head(kinds), at(100, 7));
            assert_eq!(log.floor(kinds), at(100, 7));
            assert!(
                log.holds(kinds, at(100, 7)),
                "its place, though no event is retained"
            );
            assert!(!log.holds(kinds, at(100, 4)));
            assert!(
                !log.holds(kinds, at(3, 4)),
                "what stood before the restart is gone"
            );
            assert!(!log.holds(kinds, at(0, 0)));
        }
        let mut out = Vec::new();
        assert_eq!(log.read(All, at(100, 7), 8, &mut out), Ok(()));
        assert!(out.is_empty());
        assert_eq!(log.read(All, at(99, 7), 8, &mut out), Err(Lost));
        assert_eq!(
            log.read(All, at(4, 4), 8, &mut out),
            Err(Lost),
            "a reader at the head before the restart"
        );
        assert_eq!(
            log.read(All, at(100, 4), 8, &mut out),
            Err(Lost),
            "the same position of another history"
        );
        apply(&log, at(101, 7), 9);
        assert_eq!(log.read(All, at(100, 7), 8, &mut out), Ok(()));
        assert_eq!(marks(&out), [9]);
        assert_eq!(out.first().map(|entry| entry.at), Some(at(101, 7)));
        assert_eq!(
            log.head(Watched),
            at(100, 7),
            "no membership write since the restart"
        );
    }

    #[test]
    fn a_store_that_follows_writes_under_no_term_until_it_takes_one() {
        // The store leads term 3. A leader's event of term 5 lands by the
        // store's own write or by the mark, term 5 is a snapshot's, or the
        // store is told term 5 under the guard of its first position.
        for way in ["a write", "a mark", "a restart", "the term told"] {
            let log = Log::new(8, 8);
            log.term_rose(3);
            load(&log, 1);
            assert_eq!(log.head(All), at(2, 3), "its own write while it leads");
            match way {
                "a write" => apply(&log, at(3, 5), 7),
                "a mark" => log.adopt(at(3, 5)).expect("the next position").mark(),
                "a restart" => log.restart(at(3, 5)),
                _ => {
                    let adopting = log.adopt(at(3, 5)).expect("the next position");
                    log.term_rose(5);
                    assert!(adopting.landed());
                }
            }
            assert_eq!((log.head(All), log.term()), (at(3, 5), 5), "{way}");
            // A late write of its own: the term it led is over.
            load(&log, 2);
            assert_eq!(log.head(All), at(4, 0), "{way}");
            assert!(!log.holds(All, at(4, 0)), "{way}");
            assert!(!log.holds(All, at(4, 3)), "{way}");
            // No leader's history continues from it.
            assert_eq!(log.adopt(at(5, 5)).err(), Some(Lost), "{way}");
            // A take is above every term the log followed.
            log.term_rose(5);
            log.term_rose(4);
            assert_eq!(log.head(All), at(4, 0), "{way}: no take at term 5");
            log.term_rose(6);
            load(&log, 3);
            assert_eq!(log.head(All), at(6, 6), "{way}: it leads again");
        }
    }

    #[test]
    fn a_store_that_leads_applies_no_event_under_its_own_term() {
        let log = Log::new(8, 8);
        log.term_rose(3);
        load(&log, 1);
        // Term 3 is this store's: an event under it is no leader's.
        assert_eq!(log.adopt(at(3, 3)).err(), Some(Lost));
        // A later term's event lands, and the store follows that term on.
        apply(&log, at(3, 4), 7);
        apply(&log, at(4, 4), 8);
        assert_eq!(log.head(All), at(4, 4));
    }

    #[test]
    fn a_restored_leaders_late_write_takes_no_place_its_term_wrote_before() {
        // Three replicas. The first leads term 1 through position 4. One
        // follower applied all four; the other applied two and took term 2
        // at position 3: the shorter history.
        let (first, ahead, behind) = (Log::new(8, 8), Log::new(8, 8), Log::new(8, 8));
        first.term_rose(1);
        for mark in 2..=4 {
            load(&first, mark);
        }
        let led = all(&first);
        applies(&ahead, &led);
        applies(&behind, &led[..2]);
        behind.term_rose(2);
        assert_eq!(behind.head(All), at(3, 2));
        // The first, deposed, is not held there. It is restored from that
        // leader's snapshot, and one write of its own lands a moment later.
        assert!(!behind.holds(All, first.head(All)));
        first.restart(behind.head(All));
        load(&first, 99);
        assert_eq!(
            first.head(All),
            at(4, 0),
            "term 1 wrote position 4 before, in another history"
        );
        // The replica that is ahead leads next and never followed term 2:
        // position 4 under term 1 is its own history's.
        ahead.term_rose(3);
        assert!(ahead.holds(All, at(4, 1)));
        assert!(!ahead.holds(All, first.head(All)));
        assert_eq!(
            first.adopt(at(5, 3)).err(),
            Some(Lost),
            "it starts again from a snapshot"
        );
    }

    #[test]
    fn a_term_told_while_its_start_is_applied_is_the_leaders_and_no_take() {
        let log = Log::new(8, 8);
        apply(&log, at(1, 3), 1);
        // The standby's store is told the leader's term by its own entry,
        // under the guard of that term's first position.
        let adopting = log.adopt(at(2, 5)).expect("the next position");
        log.term_rose(5);
        assert!(adopting.landed());
        drop(adopting);
        assert_eq!(places(&all(&log)), [at(1, 3), at(2, 5)]);
        assert_eq!(marks(&all(&log)), [1, TERM]);
        load(&log, 2);
        assert_eq!(
            log.head(All),
            at(3, 0),
            "the store follows term 5 and does not lead it"
        );
    }
}
