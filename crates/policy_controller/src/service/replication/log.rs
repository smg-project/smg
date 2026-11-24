//! The log: the last replicated writes of one store, each at its position
//! and under its term, in one ring and one order of positions.

use std::{
    collections::{vec_deque, VecDeque},
    fmt, mem,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use parking_lot::Mutex;

use crate::store::{Journal, MembershipChange, Position, WorkerRow};

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
    /// The store's term rose: the first position of a term, one of the
    /// watched kinds. It changes nothing a standby holds; it is why the
    /// head of a store that leads is never under term `0`, and how a
    /// follower of the log learns the term before the leader's next write.
    /// A standby journals it too at the place of a load row an older
    /// leader tells it: the row is kept nowhere, and its place stays
    /// counted.
    Term,
    /// A generation applied, by its number: one of the watched kinds, and
    /// told to no watch. The document it runs from is its door's to hold,
    /// never the log's: an apply costs the ring one small entry.
    Generation { generation: u64 },
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
pub struct LogPositionLost;

/// Why a store does not lead a term its replica took ([`Log::leads`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotLeading {
    /// The log has seen no term that high: the store is told the term next
    /// and leads it from there. A moment.
    NotYet,
    /// The store follows the term given, at or above the one asked: its
    /// take started no term in the log, or the store followed another
    /// history since. No write of its own stands under the asked term from
    /// here.
    Follows(u64),
    /// The store leads the later term given: the take that was asked about
    /// is over.
    Past(u64),
}

/// What one reader of the watched kinds keeps between its reads
/// ([`Log::read_if_moved`]): the state of those kinds its last read reached
/// the end of, or none. The reader keeps it and hands it back, nothing
/// else. It is no place of the log: a restart moves the log past it and
/// writes no entry.
#[derive(Debug, Default)]
pub struct ReadStamp(Option<u64>);

#[cfg(test)]
thread_local! {
    /// How many times this thread searched the ring ([`Kept::until`]): a
    /// case reads it to count the searches of one read.
    static SEARCHES: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// The ring: the retained events, oldest first.
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
    /// place before the oldest, which a compare answers and no search.
    /// `until` is how many retained events stand at or before the position
    /// of `at` ([`Kept::until`]), and where the ring retains the place it
    /// is the last of them.
    fn has_at(&self, at: Position, until: usize) -> bool {
        at == self.floor
            || until
                .checked_sub(1)
                .and_then(|last| self.events.get(last))
                .is_some_and(|entry| entry.at == at)
    }

    /// The retained events from the one at index `from` on, oldest first.
    fn events_from(&self, from: usize) -> vec_deque::Iter<'_, Entry> {
        self.events.range(from..)
    }

    /// How many retained events stand at or before position `seq`: the
    /// index of the first one after it, searched from the head
    /// ([`first_after`]). A reader stands near the head, and its search
    /// takes steps by its distance from there, not by the ring's size.
    fn until(&self, seq: u64) -> usize {
        #[cfg(test)]
        SEARCHES.with(|searches| searches.set(searches.get() + 1));
        first_after(self.events.len(), |index| {
            self.events
                .get(index)
                .is_some_and(|entry| entry.at.seq <= seq)
        })
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

/// The first of `len` places in a row that does not stand `before`, where
/// every place that does comes ahead of every one that does not; `len`
/// where all do. Searched from the end, by offsets that double, then by
/// halves between the last two looks: one look where every place stands
/// before or the row holds one place, two where all but the last of a
/// longer row do, and at most twice the steps of a search of the whole
/// row.
fn first_after(len: usize, mut before: impl FnMut(usize) -> bool) -> usize {
    // Every place before `lo` stands before, and none from `hi` on does.
    let (mut lo, mut hi) = (0, len);
    let mut back = 1;
    while lo < hi {
        let look = len.saturating_sub(back);
        if before(look) {
            lo = look + 1;
            break;
        }
        hi = look;
        back *= 2;
    }
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if before(mid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

struct Ring {
    /// The retained writes: membership and status writes, starts of terms,
    /// generations applied.
    watched: Kept,
    /// The last write, with its term.
    head: Position,
    /// The leader's place for the next write, while a standby applies the
    /// leader's event.
    adopting: Option<Position>,
    /// How many places a standby has asked for, wrapping: a guard gives the
    /// pending place up while its own is the last asked for, and reads the
    /// count for that alone. A guard misreads it only if it outlives 2^32
    /// later adoptions, and a guard lives one apply. Its four bytes stand
    /// beside `leads`: the count adds nothing to the ring's size.
    adoptions: u32,
    /// Whether the store leads the log's term: from its take until it
    /// follows another history.
    leads: bool,
    /// Whether no standby can follow the log ([`Log::kept_alone`]).
    alone: bool,
    /// The highest number the log took under no term. No term names the
    /// store's own line of writes, and a restart may set the head below a
    /// number that line took: a place under no term is numbered past it
    /// ([`Ring::fresh`]), so it names one write for as long as the log
    /// lives.
    unnamed: u64,
    /// The place before the store's first write: the default, or the
    /// number its run drew ([`Log::starting_after`]). A ring stands at it
    /// until it lets its first write go or the log restarts at another
    /// place, a snapshot's.
    first: Position,
}

impl Ring {
    /// Whether a reader continues from `at` ([`Ring::after`]).
    fn continues(&self, at: Position) -> bool {
        self.after(at).is_some()
    }

    /// What a reader at `at` has before it: the retained events after that
    /// place, oldest first. `None` unless the log has that place under
    /// that term and every event after it is retained. The ring is
    /// searched once: the search that says whether it has the place finds
    /// what follows the place there.
    fn after(&self, at: Position) -> Option<vec_deque::Iter<'_, Entry>> {
        if at.seq < self.watched.floor.seq {
            return None;
        }
        let from = self.watched.until(at.seq);
        self.watched
            .has_at(at, from)
            .then(|| self.watched.events_from(from))
    }

    /// The number of a place under no term that would stand at `seq`:
    /// `seq`, or the first past every number such a place took before.
    fn fresh(&mut self, seq: u64) -> u64 {
        self.unnamed = seq.max(self.unnamed + 1);
        self.unnamed
    }

    /// Starts the places after `first`, under no term: the head, the
    /// ring's floor and the first place. The store's own line
    /// takes the number too and keeps a higher one it stood at: no write
    /// under no term is numbered at a place the log said was its first.
    fn start_after(&mut self, first: u64) {
        let at = Position {
            seq: first,
            term: 0,
        };
        self.watched.restart(at);
        self.head = at;
        self.first = at;
        self.unnamed = self.unnamed.max(first);
    }
}

/// The log of one store: its last replicated writes in one order of
/// positions, in one ring. They are the watched kinds (membership and
/// status writes, the starts of terms, the generations applied): a reader
/// keeps its place for as long as the ring holds what follows it.
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
/// nothing else. A write is told to one listener once its store has let its
/// locks go ([`Log::on_watched`]).
///
/// The bound in bytes: a slot is at most 120 bytes whatever the event. A
/// membership event's rows stand apart from its slot, 352 bytes a
/// row and its strings: about 0.55 KB an event at a row of a short URL, one
/// model, no label, no pairing descriptor and no encode connector, a second
/// row for a replacement, so 4,096 membership events are about 2.2 MB at
/// that row. A pairing descriptor stands apart from its row, 168 bytes and
/// its strings, held once for every copy of that row, and so does a
/// provider's key, 64 bytes and its strings.
///
/// The fields stand in the order written (`repr(C)`), the lock first: a
/// read addresses the ring off the log's own pointer, whatever the ring's
/// size. On a 64-bit target each field's size is a multiple of eight
/// bytes, so the order costs no padding. The build holds both.
#[repr(C)]
pub struct Log {
    ring: Mutex<Ring>,
    /// The highest term the log has seen: the store's takes, the events it
    /// applied, the place it restarted at.
    term: AtomicU64,
    /// How many events are retained.
    watched: usize,
    /// Told that the watched kinds moved, once a write of them is released.
    told: Option<Box<dyn Fn() + Send + Sync>>,
    /// How often the watched kinds moved: a write of them, a mark, a
    /// restart. It moves under the log's lock, and a reader that keeps a
    /// [`ReadStamp`] loads it under none.
    moves: AtomicU64,
    /// The highest count of moves told: `told` is owed a telling while
    /// `moves` is past it, and is told under no lock.
    told_at: AtomicU64,
}

// The layout the log's doc states: the lock is the first field, and on a
// 64-bit target the declared order leaves no padding between the fields.
const _: () = {
    assert!(mem::offset_of!(Log, ring) == 0);
    #[cfg(target_pointer_width = "64")]
    assert!(
        size_of::<Log>()
            == size_of::<Mutex<Ring>>()
                + 3 * size_of::<AtomicU64>()
                + size_of::<usize>()
                + size_of::<Option<Box<dyn Fn() + Send + Sync>>>()
    );
};

impl fmt::Debug for Log {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Log")
            .field("head", &self.head())
            .field("floor", &self.floor())
            .field("watched", &self.watched)
            .finish_non_exhaustive()
    }
}

impl Log {
    /// The most entries one read copies out, whatever its reader asks: the
    /// longest a reader holds the log's lock.
    pub const BATCH: usize = 256;

    /// A log that retains the last `watched` writes (at least one). It
    /// carries no load row: a store that journals to it pays nothing at a
    /// load write.
    pub fn new(watched: usize) -> Self {
        let watched = watched.max(1);
        Self {
            ring: Mutex::new(Ring {
                watched: Kept::of(watched),
                head: Position::default(),
                adopting: None,
                adoptions: 0,
                leads: false,
                alone: false,
                unnamed: 0,
                first: Position::default(),
            }),
            term: AtomicU64::new(0),
            watched,
            told: None,
            moves: AtomicU64::new(0),
            told_at: AtomicU64::new(0),
        }
    }

    /// Calls `told` after a write of the watched kinds and after a
    /// restart: how a reader of those kinds learns the log moved without
    /// polling it. `told` runs under no lock of the log or of the store: a
    /// store's write is told once the store has let its locks go
    /// ([`Journal::released`]), so what `told` wakes holds no writer back.
    /// It is handed no place, so a telling takes no lock of the log: its
    /// reader reads the head itself ([`Log::head`]), which stands at or
    /// after each write told. Writes that landed together may be told
    /// once.
    #[must_use]
    pub fn on_watched(mut self, told: impl Fn() + Send + Sync + 'static) -> Self {
        self.told = Some(Box::new(told));
        self
    }

    /// The log with its places starting after `first`: the number a run
    /// draws, so that a place of another run names no write of this one.
    /// The head is that place under no term, and the next
    /// write follows it: from there the log is its store's whole history
    /// ([`Log::first`]). The number is one of the store's own line: after
    /// a restart below it, a write under no term is numbered past it.
    #[must_use]
    pub fn starting_after(mut self, first: u64) -> Self {
        self.ring.get_mut().start_after(first);
        self
    }

    /// The log of a replica whose lease lives in its own process. No other
    /// replica can see that lease, so no standby follows the log
    /// ([`Log::alone`]).
    #[must_use]
    pub fn kept_alone(mut self) -> Self {
        self.ring.get_mut().alone = true;
        self
    }

    /// Whether no standby can follow the log ([`Log::kept_alone`]).
    pub fn alone(&self) -> bool {
        self.ring.lock().alone
    }

    /// The last write, with its term; the default before any. It is where
    /// a cut of the rows stands: the rows are as that write left them.
    pub fn head(&self) -> Position {
        self.ring.lock().head
    }

    /// The earliest place a reader continues from: every event after it is
    /// retained.
    pub fn floor(&self) -> Position {
        self.ring.lock().watched.floor
    }

    /// The place before the store's first write, where a reader that holds
    /// nothing joins the log: while the log is its store's whole history
    /// and retains every write. `None` once the ring let its first write
    /// go, and after a restart at another place (what the store held at a
    /// snapshot is no event here): that reader starts from a snapshot.
    pub fn first(&self) -> Option<Position> {
        let ring = self.ring.lock();
        let first = ring.first;
        (ring.watched.floor == first).then_some(first)
    }

    /// The highest term the log has seen: the store's last take, or a later
    /// term it applied an event of or restarted at. The store's own writes
    /// are under it while the store leads, and under no term otherwise.
    pub fn term(&self) -> u64 {
        self.term.load(Ordering::Relaxed)
    }

    /// The term the store followed, `None` while the store leads the log's
    /// term and its own writes stand under it. The store follows before
    /// its first take (term `0`) and once it applied an event or restarted
    /// at a snapshot. A take at or below a term the log has seen starts no
    /// lead and leaves that term here: the store writes under no term
    /// until it takes a later one.
    pub fn followed(&self) -> Option<u64> {
        let ring = self.ring.lock();
        (!ring.leads).then(|| self.term.load(Ordering::Relaxed))
    }

    /// Whether the store leads `term`, in one read under the log's lock:
    /// the log's term is `term` and the store's own writes stand under it.
    /// A refusal carries the term the log stands at, so no caller reads the
    /// log again for its words. Three callers read it, each in its own
    /// turn: the service's guard, the door's mint of a worker's life and
    /// the replication's check.
    pub fn leads(&self, term: u64) -> Result<(), NotLeading> {
        let ring = self.ring.lock();
        let seen = self.term.load(Ordering::Relaxed);
        if seen < term {
            Err(NotLeading::NotYet)
        } else if !ring.leads {
            Err(NotLeading::Follows(seen))
        } else if seen == term {
            Ok(())
        } else {
            Err(NotLeading::Past(seen))
        }
    }

    /// Whether a reader that another store's history left at `at`
    /// continues from it: the log has that position under that term, so
    /// the two share the history before it (the replication contract's
    /// rejoin rule), and every event after it is retained. The log has a
    /// place while the ring retains its event, and as the place before the
    /// ring's oldest event, which is the place before every write until
    /// the ring lets its first go. Refused: another term at the position,
    /// a position past the head or one the log let go, and a position that
    /// names no term (nobody led term `0`: two stores may each hold a write
    /// of their own there, and so may one store before and after its
    /// process restarts, since a number under no term lives as long as the
    /// log and no longer).
    pub fn holds(&self, at: Position) -> bool {
        (at.term != 0 || at.seq == 0) && self.ring.lock().continues(at)
    }

    /// The writes after `after`, oldest first, into `out` (cleared first):
    /// at most `at_most` and never more than [`Log::BATCH`], and all under
    /// one term, so a batch splits where the term changed. Nothing when
    /// none followed.
    ///
    /// `after` is where the reader stands: the place [`Log::holds`]
    /// accepted or a cut's, then the `at` of the last entry it read.
    /// [`LogPositionLost`] unless the log has that position under that term and
    /// retains every event after it: whatever befell the log between two
    /// reads (the ring let the place go, the log restarted at a snapshot),
    /// a reader is served the history it stands in or nothing.
    pub fn read(
        &self,
        after: Position,
        at_most: usize,
        out: &mut Vec<Entry>,
    ) -> Result<(), LogPositionLost> {
        self.batch(after, at_most, out)?;
        Ok(())
    }

    /// [`Log::read`] for a reader that comes back whether or not the log
    /// moved, a watch's stream. `stamp` is what that one reader keeps
    /// between its reads, and `after` is where it stands. While the log
    /// stands as the read that set the stamp left it, nothing is read
    /// (`out` is cleared) and no lock of the log is taken: the reader was
    /// handed every entry after its place.
    ///
    /// A read sets the stamp where it reached the end of the log, under
    /// the lock every move takes: a write, a mark and a restart each move
    /// the log past it. A read that stopped short of the end, at a full
    /// batch or where the term changed, sets none, and the next reads on.
    /// Whether the reader's place is still the log's is asked where the
    /// log is read: a reader that is [`LogPositionLost`] hears it at its
    /// first read after a move.
    pub fn read_if_moved(
        &self,
        stamp: &mut ReadStamp,
        after: Position,
        at_most: usize,
        out: &mut Vec<Entry>,
    ) -> Result<(), LogPositionLost> {
        if stamp.0 == Some(self.moves.load(Ordering::Acquire)) {
            out.clear();
            return Ok(());
        }
        stamp.0 = self.batch(after, at_most, out)?;
        Ok(())
    }

    /// One batch for [`Log::read`] and [`Log::read_if_moved`]. Where it
    /// reached the end of what the log holds, it says how often the log
    /// had moved by then: a [`ReadStamp`] of the state it read.
    fn batch(
        &self,
        after: Position,
        at_most: usize,
        out: &mut Vec<Entry>,
    ) -> Result<Option<u64>, LogPositionLost> {
        out.clear();
        let at_most = at_most.min(Self::BATCH);
        let ring = self.ring.lock();
        let Some(mut events) = ring.after(after) else {
            return Err(LogPositionLost);
        };
        let mut batch_term = None;
        let mut ended = false;
        while out.len() < at_most {
            let Some(entry) = events.next() else {
                ended = true;
                break;
            };
            if batch_term.is_some_and(|of_batch| of_batch != entry.at.term) {
                break;
            }
            batch_term = Some(entry.at.term);
            out.push(entry.clone());
        }
        // Loaded under the lock still, which every move takes: the count
        // names the state this batch read.
        Ok(ended.then(|| self.moves.load(Ordering::Relaxed)))
    }

    /// One write journaled. `by` is the place a standby's applier holds
    /// for the write ([`Log::adopt`]), and none for a write of the store's
    /// own. The applier's write lands at its place while its guard holds
    /// it, and at no other place of its leader's. A write of the store's
    /// own takes no such place: it stands where the store's own writes
    /// stand, and the hold ends with it, so the leader's event does not
    /// land.
    ///
    /// An applier's write whose hold ended is let go while the store
    /// follows: a snapshot puts its tables right. Where the store took a
    /// term since the place was asked for, its tables took the row under
    /// its own lead and no snapshot follows: the row stands as a write of
    /// the store's own under that term, so the log holds what the tables
    /// hold. A term's start wrote no row, and is let go there too.
    fn push(&self, by: Option<Position>, event: Event) {
        let mut ring = self.ring.lock();
        let held = ring.adopting.take();
        let lands = by.filter(|at| held == Some(*at));
        // The hold ended before the applier's write: it is let go, or it
        // stands as the store's own.
        if let (Some(at), None) = (by, lands) {
            if !self.took_since(&ring, at) || matches!(event, Event::Term) {
                return;
            }
        }
        let at = match lands {
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
                    // A generation applied under no term takes no place:
                    // the take that starts a term tells the standing one.
                    if matches!(event, Event::Generation { .. }) {
                        return;
                    }
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

    /// Whether the store took a term since the place `at` was asked for:
    /// it leads a term above the place's. A guard is refused a place under
    /// a term below the log's ([`Log::adopt`]), so the log's term rose
    /// past it since, and by a take: the store leads.
    fn took_since(&self, ring: &Ring, at: Position) -> bool {
        ring.leads && at.term < self.term.load(Ordering::Relaxed)
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
        ring.watched.push(self.watched, Entry { at, event });
        self.moved(ring);
    }

    /// Notes that the watched kinds moved: a write of them, a mark, a
    /// restart. `_held` is the log's lock, held: the count moves under it,
    /// so a batch read under that lock names the state it read. The move
    /// is told under no lock, and the count alone owes it ([`Log::tell`]).
    fn moved(&self, _held: &Ring) {
        self.moves.fetch_add(1, Ordering::Release);
    }

    /// Tells the listener that the watched kinds moved since its last
    /// telling, if they did. Called under no lock of the log or of the
    /// store: once a store has let a write's locks go, and at the end of a
    /// restart, of a mark and of a join.
    ///
    /// A telling is owed while the count of moves is past the highest
    /// count told. The count is loaded before the kept one is raised to
    /// it, so a move is told after its count moved, and the reader that
    /// telling wakes finds the log past its [`ReadStamp`], unless it read
    /// past that move already, whatever brought it to read: its stamp
    /// stands then, and it reads nothing until a later move, which owes
    /// a telling of its own. Of the releases that race, each that raises
    /// the kept count tells: no move goes untold, and a release that
    /// raises nothing tells nothing.
    fn tell(&self) {
        let Some(told) = &self.told else {
            return;
        };
        let moves = self.moves.load(Ordering::Acquire);
        if self.told_at.fetch_max(moves, Ordering::AcqRel) < moves {
            told();
        }
    }

    /// The log's lock, held: for a case that shows what waits for it.
    #[cfg(test)]
    pub(crate) fn held(&self) -> impl Sized + '_ {
        self.ring.lock()
    }

    /// A standby's application of the leader's event at `at`: while the
    /// returned guard stands, the write that names that place is journaled
    /// there under that term, the leader's own, so a position and its term
    /// name one history on every replica that holds them. The applier's
    /// write names it, through the store's door for a followed write
    /// ([`Journal::member`]), and no other write does. Refused
    /// unless `at` is the position after the head, under a named term not
    /// below any the log has seen, and the head is a leader's write or the
    /// start: a gap, a replay, a falling term and a write of the store's
    /// own under no term are not where a leader's history continues. Nor
    /// is an event under the term this store leads: taken once, that term
    /// has one writer. A log that stands at the place its run drew takes
    /// a leader's first position by [`Log::join`].
    ///
    /// A store that follows has one writer of journaled writes, the one
    /// that applies the leader's batches. A write of the store's own that
    /// comes while a guard stands has no part in the guard's place: it
    /// stands under no term, the hold ends with it, and the event does not
    /// land ([`AdoptionGuard::landed`]). Once an event lands, the term the
    /// store led is over: a write of its own stands under no term until the
    /// store takes a term again.
    pub fn adopt(&self, at: Position) -> Result<AdoptionGuard<'_>, LogPositionLost> {
        let mut ring = self.ring.lock();
        let unled = ring.head.term == 0 && ring.head.seq != 0;
        let term = self.term.load(Ordering::Relaxed);
        let own = ring.leads && at.term == term;
        if unled || own || at.term == 0 || at.seq != ring.head.seq + 1 || at.term < term {
            return Err(LogPositionLost);
        }
        ring.adopting = Some(at);
        ring.adoptions = ring.adoptions.wrapping_add(1);
        Ok(AdoptionGuard {
            log: self,
            at,
            nth: ring.adoptions,
        })
    }

    /// Joins the history of the leader this store follows from its start.
    /// `at` is that history's first position, the start of the leader's
    /// first term: where a leader continues a standby that applied
    /// nothing. The log still stands at the place its own run drew, which
    /// is no start a guard takes ([`Log::adopt`]): it starts after the
    /// place before `at` instead, as its leader's did
    /// ([`Log::starting_after`]), and journals the position, which writes
    /// no row. From here it is its leader's history from the first place
    /// on, and says that place as its own ([`Log::first`]): the store
    /// serves a reader that holds nothing from there when it leads. It
    /// follows the term from this step, so a take of that term again
    /// starts no lead in it. The first place's number and the one its own
    /// run drew both stay on the store's own line: a write of its own under
    /// no term is numbered past the two. The listener is told at its end,
    /// under no lock of the log.
    ///
    /// Refused once the log holds a write or restarted at a snapshot, and
    /// for a position under no term or with no place before it. The log is
    /// then as it was.
    pub fn join(&self, at: Position) -> Result<(), LogPositionLost> {
        let mut ring = self.ring.lock();
        let Some(first) = at.seq.checked_sub(1) else {
            return Err(LogPositionLost);
        };
        if ring.head != ring.first || at.term == 0 {
            return Err(LogPositionLost);
        }
        ring.start_after(first);
        ring.adopting = None;
        self.follow(&mut ring, at);
        self.append(&mut ring, at, Event::Term);
        drop(ring);
        self.tell();
        Ok(())
    }

    /// Start again at `at`, the position of a snapshot this store was
    /// restored from: nothing is retained, the head is `at`,
    /// and the next write follows it. A reader that stood anywhere else is
    /// [`LogPositionLost`] at its next read, and the listener is told. The
    /// store follows the snapshot's leader from here: the term it led is over,
    /// so a late write of its own takes no position that term wrote before the
    /// restart, and none a write of its own took under no term: it is numbered
    /// past both.
    ///
    /// Refused as an event of that term would be ([`Log::adopt`]): a place
    /// under no term, a term below the highest the log has seen, and the
    /// term the store leads. The log is then as it was.
    pub fn restart(&self, at: Position) -> Result<(), LogPositionLost> {
        self.restart_untold(at)?;
        self.tell();
        Ok(())
    }

    /// [`Log::restart`] with its telling left out: for a store that
    /// restores inside its own locks and tells at their release
    /// ([`Journal::released`]).
    fn restart_untold(&self, at: Position) -> Result<(), LogPositionLost> {
        let mut ring = self.ring.lock();
        let term = self.term.load(Ordering::Relaxed);
        if at.term == 0 || at.term < term || (ring.leads && at.term == term) {
            return Err(LogPositionLost);
        }
        ring.watched.restart(at);
        ring.head = at;
        ring.adopting = None;
        self.follow(&mut ring, at);
        self.moved(&ring);
        Ok(())
    }
}

/// One leader's event being applied to a standby's store ([`Log::adopt`]).
/// The place is this guard's own to give up: an applier that keeps one
/// binding for its guards makes the next guard before the last one drops,
/// and the last one leaves the next one's place.
#[derive(Debug)]
#[must_use = "the place is given up when the guard drops"]
pub struct AdoptionGuard<'a> {
    log: &'a Log,
    at: Position,
    /// Which of the log's adoptions this is.
    nth: u32,
}

impl AdoptionGuard<'_> {
    /// Whether the applier's write landed: the head is the adopted
    /// position. A term's start writes no row, and the store's door for it
    /// journals the place alone ([`Journal::term_followed`]). A store that
    /// journaled nothing there is not the leader's twin: its tables did
    /// not take the event, or a write of its own ended the hold first.
    pub fn landed(&self) -> bool {
        self.log.head() == self.at
    }

    /// Applies a leader's event that writes nothing to this store, the
    /// first position of a term: the place alone is journaled, and the next
    /// event follows it. A standby marks the place of a load row too: its
    /// store keeps no load row of its leader's. The listener is told at
    /// its end, under no lock of the log.
    pub fn mark(&self) {
        let mut ring = self.log.ring.lock();
        if let Some(at) = self.pending(&mut ring) {
            self.log.follow(&mut ring, at);
            self.log.append(&mut ring, at, Event::Term);
        }
        drop(ring);
        self.log.tell();
    }

    /// The place this guard asked for, taken from the log: none once a
    /// write took it or a later guard asked for its own.
    fn pending(&self, ring: &mut Ring) -> Option<Position> {
        if ring.adoptions == self.nth {
            ring.adopting.take()
        } else {
            None
        }
    }
}

impl Drop for AdoptionGuard<'_> {
    fn drop(&mut self) {
        // A place no write took is not kept for a later one.
        let mut ring = self.log.ring.lock();
        self.pending(&mut ring);
    }
}

impl Journal for Log {
    fn member(
        &self,
        at: Option<Position>,
        epoch: u64,
        change: MembershipChange,
        row: Option<&WorkerRow>,
        left: Option<&WorkerRow>,
    ) {
        // The rows are copied before the log's lock is taken.
        let copy = |row: &WorkerRow| Arc::new(row.clone());
        self.push(
            at,
            Event::Member {
                epoch,
                change,
                row: row.map(copy),
                left: left.map(copy),
            },
        );
    }

    fn generation(&self, at: Option<Position>, generation: u64) {
        self.push(at, Event::Generation { generation });
    }

    fn term_rose(&self, term: u64) {
        let mut ring = self.ring.lock();
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

    fn term_followed(&self, at: Position) {
        self.push(Some(at), Event::Term);
    }

    fn head(&self) -> Position {
        Log::head(self)
    }

    fn followed(&self) -> Option<u64> {
        Log::followed(self)
    }

    fn released(&self) {
        self.tell();
    }

    fn restart(&self, at: Position) -> bool {
        self.restart_untold(at).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{atomic::AtomicUsize, mpsc},
        thread,
        time::Duration,
    };

    use super::*;
    use crate::{
        engine::view::WorkerId, service::replication::tests::ascends, store::WorkerStatus,
    };

    /// The mark a term's first position reads as.
    const TERM: u32 = u32::MAX;
    /// The mark a membership event reads as, above its epoch.
    const MEMBER: u32 = 1_000_000;
    /// A generation's mark: its number above this.
    const GENERATION: u32 = 2_000_000;

    /// A status write of the store's own, marked by its epoch.
    fn status(log: &Log, epoch: u32) {
        log.member(None, u64::from(epoch), drained(epoch), None, None);
    }

    /// A leader's status write applied at `at`, the place a guard holds.
    fn status_at(log: &Log, at: Position, epoch: u32) {
        log.member(Some(at), u64::from(epoch), drained(epoch), None, None);
    }

    fn drained(epoch: u32) -> MembershipChange {
        MembershipChange::StatusChanged {
            id: WorkerId(u64::from(epoch)),
            from: WorkerStatus::Ready,
            to: WorkerStatus::Draining,
        }
    }

    /// A membership write of the store's own, marked by its epoch above
    /// [`MEMBER`].
    fn member(log: &Log, epoch: u32) {
        log.member(None, u64::from(epoch), relabel(epoch), None, None);
    }

    /// A leader's membership write applied at `at`, the place a guard
    /// holds.
    fn member_at(log: &Log, at: Position, epoch: u32) {
        log.member(Some(at), u64::from(epoch), relabel(epoch), None, None);
    }

    fn relabel(epoch: u32) -> MembershipChange {
        MembershipChange::Relabelled {
            id: WorkerId(u64::from(epoch)),
        }
    }

    /// Each event's mark, in order: a status write's epoch, another
    /// membership event's epoch above [`MEMBER`], [`TERM`] for a term's
    /// start.
    fn marks(entries: &[Entry]) -> Vec<u32> {
        entries
            .iter()
            .map(|entry| match &entry.event {
                Event::Member { epoch, change, .. } => {
                    let epoch = u32::try_from(*epoch).expect("a test's epoch");
                    if matches!(change, MembershipChange::StatusChanged { .. }) {
                        epoch
                    } else {
                        MEMBER + epoch
                    }
                }
                Event::Term => TERM,
                Event::Generation { generation } => {
                    GENERATION + u32::try_from(*generation).expect("a test's generation")
                }
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
        all_after(log, Position::default())
    }

    /// Every entry after `after`, each batch read from the place of the
    /// last entry before it. The walk fails where a batch does not ascend
    /// from that place or where it passes what the ring holds: a read that
    /// handed the place back would append without end.
    fn all_after(log: &Log, after: Position) -> Vec<Entry> {
        let mut entries: Vec<Entry> = Vec::new();
        let mut out = Vec::new();
        loop {
            let from = entries.last().map_or(after, |entry| entry.at);
            if log.read(from, 3, &mut out) != Ok(()) || out.is_empty() {
                return entries;
            }
            ascends(from, &out);
            entries.append(&mut out);
            assert!(
                entries.len() <= log.watched,
                "a walk of more than the ring holds"
            );
        }
    }

    /// Applies a leader's status write at `at`, as a standby does.
    fn apply(log: &Log, at: Position, mark: u32) {
        let adopting = log.adopt(at).expect("the next position");
        status_at(log, at, mark);
        assert!(adopting.landed());
    }

    /// Applies a leader's entries in order, as a standby does: each by the
    /// journal's door for a followed write, at the place its guard holds.
    fn applies(follower: &Log, led: &[Entry]) {
        for entry in led {
            let adopting = follower.adopt(entry.at).expect("the next position");
            assert!(!adopting.landed());
            match &entry.event {
                Event::Term => follower.term_followed(entry.at),
                Event::Generation { generation } => {
                    follower.generation(Some(entry.at), *generation);
                }
                Event::Member {
                    epoch,
                    change,
                    row,
                    left,
                } => follower.member(
                    Some(entry.at),
                    *epoch,
                    *change,
                    row.as_deref(),
                    left.as_deref(),
                ),
            }
            assert!(adopting.landed());
        }
    }

    #[test]
    fn a_slot_is_at_most_120_bytes_and_a_membership_row_stands_apart() {
        // The log's bound in bytes is told from these: a ring's slots, and
        // apart from its slot a membership event's rows (two for a
        // replacement), each with its strings. A provider's key stands
        // apart from its row, behind a pointer's two counts.
        assert!(size_of::<Entry>() <= 120);
        assert!(size_of::<WorkerRow>() <= 352);
        assert!(size_of::<crate::store::ProviderKey>() + 2 * size_of::<usize>() <= 64);
    }

    #[test]
    fn one_count_of_positions_runs_through_the_log_and_a_term_starts_with_its_own() {
        let log = Log::new(16);
        assert_eq!(log.head(), Position::default());
        assert_eq!(log.floor(), Position::default());
        status(&log, 10);
        log.term_rose(3);
        assert_eq!(
            log.head(),
            at(2, 3),
            "the take is the term's first position"
        );
        status(&log, 11);
        member(&log, 1);
        status(&log, 12);
        assert_eq!(log.head(), at(5, 3));
        let mut out = Vec::new();
        assert_eq!(log.read(at(0, 0), 8, &mut out), Ok(()));
        assert_eq!(
            places(&out),
            [at(1, 0)],
            "a batch stops where the term changes"
        );
        assert_eq!(log.read(at(1, 0), 8, &mut out), Ok(()));
        assert_eq!(places(&out), [at(2, 3), at(3, 3), at(4, 3), at(5, 3)]);
        assert_eq!(
            marks(&out),
            [TERM, 11, MEMBER + 1, 12],
            "the writes in the log's order"
        );
    }

    #[test]
    fn a_term_never_falls_and_only_a_rise_takes_a_position() {
        let log = Log::new(16);
        log.term_rose(5);
        log.term_rose(2);
        log.term_rose(5);
        assert_eq!(log.term(), 5);
        assert_eq!(log.head(), at(1, 5), "one rise, one position");
        status(&log, 1);
        assert_eq!(log.head(), at(2, 5));
    }

    #[test]
    fn a_reader_continues_only_from_a_place_the_log_has_under_its_term() {
        // A store that wrote before any lease: the position stands under
        // term 0, which names no term. A standby does not continue from it;
        // the log's own reader, which read it here, does.
        let unled = Log::new(8);
        status(&unled, 1);
        assert_eq!(unled.head(), at(1, 0));
        assert!(!unled.holds(at(1, 0)), "had, but under no term");
        assert!(unled.holds(at(0, 0)));
        let mut out = Vec::new();
        assert_eq!(unled.read(at(1, 0), 4, &mut out), Ok(()));

        let log = Log::new(4);
        assert!(log.holds(at(0, 0)), "an empty log is joined at its start");
        assert!(!log.holds(at(1, 1)), "past the head of an empty log");
        log.term_rose(2);
        status(&log, 1);
        status(&log, 2);
        log.term_rose(3);
        assert_eq!(places(&all(&log)), [at(1, 2), at(2, 2), at(3, 2), at(4, 3)]);
        assert!(log.holds(at(0, 0)), "no first write left the ring");
        assert!(log.holds(at(3, 2)), "a status write's place");
        assert!(log.holds(at(4, 3)), "a term's first place");
        assert!(!log.holds(at(3, 3)), "had under another term");
        assert!(!log.holds(at(4, 2)), "had under another term");
        assert!(!log.holds(at(5, 3)), "past the head");
        assert!(!log.holds(at(3, 0)), "a position that names no term");
        assert!(!log.holds(at(0, 2)), "a term with no position");
        // The ring of four takes three more and lets its first three writes
        // go: the last of them is the place before the ring's oldest, under
        // its own term.
        for epoch in 1..=3 {
            member(&log, epoch);
        }
        assert_eq!(log.floor(), at(3, 2));
        assert!(!log.holds(at(0, 0)), "the first write left the ring");
        assert!(log.holds(at(3, 2)), "the place before the oldest retained");
        assert!(!log.holds(at(3, 3)), "that place under another term");
        member(&log, 4);
        assert_eq!(log.floor(), at(4, 3));
        assert!(!log.holds(at(3, 2)), "behind the ring");
        assert!(log.holds(at(4, 3)));
    }

    #[test]
    fn a_read_is_bounded_and_a_reader_at_the_head_reads_nothing() {
        let log = Log::new(20);
        for mark in 0..10 {
            status(&log, mark);
        }
        let mut out = Vec::new();
        assert_eq!(log.read(at(0, 0), 4, &mut out), Ok(()));
        assert_eq!(marks(&out), [0, 1, 2, 3]);
        assert_eq!(log.read(at(4, 0), 100, &mut out), Ok(()));
        assert_eq!(marks(&out), [4, 5, 6, 7, 8, 9]);
        assert_eq!(log.read(at(10, 0), 4, &mut out), Ok(()));
        assert!(out.is_empty(), "the reader is at the head");
        assert_eq!(
            log.read(at(11, 0), 4, &mut out),
            Err(LogPositionLost),
            "past the head"
        );
        assert_eq!(
            log.read(at(4, 1), 4, &mut out),
            Err(LogPositionLost),
            "a place the log has under another term"
        );
        assert_eq!(log.read(at(4, 0), 0, &mut out), Ok(()));
        assert!(out.is_empty(), "a read of nothing says the place stands");
        // Whatever its reader asks, one read copies the log's own bound.
        let log = Log::new(2 * Log::BATCH);
        for mark in 0..300 {
            status(&log, mark);
        }
        assert_eq!(log.read(at(0, 0), usize::MAX, &mut out), Ok(()));
        assert_eq!(out.len(), Log::BATCH);
    }

    #[test]
    fn a_ring_keeps_its_last_writes_and_a_reader_behind_it_is_lost() {
        let log = Log::new(4);
        for mark in 0..10 {
            status(&log, mark);
        }
        assert_eq!(log.head(), at(10, 0));
        assert_eq!(log.floor(), at(6, 0));
        let mut out = vec![Entry {
            at: at(99, 99),
            event: Event::Term,
        }];
        assert_eq!(log.read(at(5, 0), 8, &mut out), Err(LogPositionLost));
        assert!(
            out.is_empty(),
            "a lost read leaves nothing of an earlier one"
        );
        // The reader that applied position 6 reads the oldest retained next.
        assert_eq!(log.read(at(6, 0), 8, &mut out), Ok(()));
        assert_eq!(marks(&out), [6, 7, 8, 9]);
        assert_eq!(out.first().map(|entry| entry.at), Some(at(7, 0)));
        // A log asked for no slot keeps one.
        let log = Log::new(0);
        member(&log, 1);
        member(&log, 2);
        assert_eq!(log.floor(), at(1, 0));
        assert_eq!(marks(&all_after(&log, at(1, 0))), [MEMBER + 2]);
    }

    #[test]
    fn writers_on_many_threads_leave_dense_positions_under_rising_terms() {
        const WRITERS: u32 = 4;
        const EACH: u32 = 500;
        const TERMS: u32 = EACH / 50;
        let log = Arc::new(Log::new((WRITERS * EACH + TERMS) as usize));
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
                            status(&log, writer * EACH + write);
                        }
                    }
                });
            }
        });
        // Every writer asks for the same ten terms: each rises once.
        let total = u64::from(WRITERS * EACH + TERMS);
        assert_eq!(log.head().seq, total);
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
    }

    #[test]
    fn the_log_tells_a_write_once_it_is_released() {
        let told = Arc::new(AtomicUsize::new(0));
        let log = {
            let told = Arc::clone(&told);
            Log::new(16).on_watched(move || {
                told.fetch_add(1, Ordering::Relaxed);
            })
        };
        // How often the listener was told, beside the head it would read.
        let told = || (told.load(Ordering::Relaxed), log.head());
        // Nothing is told inside a write: the store that wrote says when
        // its locks are let go.
        status(&log, 1);
        member(&log, 1);
        assert_eq!(told().0, 0, "told inside the write");
        log.released();
        assert_eq!(told(), (1, at(2, 0)));
        // Writes that landed together are told once, at the head. A release
        // after nothing new tells nothing.
        log.term_rose(2);
        member(&log, 2);
        status(&log, 2);
        log.released();
        log.released();
        assert_eq!(told(), (2, at(5, 2)));
        // A standby's log tells what it marks and where it restarts, each
        // at its own end.
        log.adopt(at(6, 3)).expect("the next position").mark();
        assert_eq!(told(), (3, at(6, 3)));
        apply(&log, at(7, 3), 3);
        log.restart(at(11, 4)).expect("a place the log takes");
        assert_eq!(told(), (4, at(11, 4)));
    }

    #[test]
    fn a_telling_takes_no_lock_of_the_log() {
        let (told, heard) = mpsc::channel();
        let log = Arc::new(Log::new(16).on_watched(move || {
            let _ = told.send(());
        }));
        member(&log, 1);
        // The log's lock is held here while another thread makes the
        // telling: a telling that reads the log would wait for it.
        let ring = log.ring.lock();
        let teller = {
            let log = Arc::clone(&log);
            thread::spawn(move || log.released())
        };
        heard
            .recv_timeout(Duration::from_secs(10))
            .expect("the telling waited for the log's lock");
        drop(ring);
        teller.join().expect("the teller");
    }

    /// A reader that keeps its stamp ([`Log::read_if_moved`]): it stands at
    /// the last entry it read.
    #[derive(Debug, Default)]
    struct Reader {
        stamp: ReadStamp,
        at: Position,
    }

    impl Reader {
        /// One read of at most `at_most` entries: their marks.
        fn reads(&mut self, log: &Log, at_most: usize) -> Result<Vec<u32>, LogPositionLost> {
            let mut out = Vec::new();
            log.read_if_moved(&mut self.stamp, self.at, at_most, &mut out)?;
            if let Some(last) = out.last() {
                self.at = last.at;
            }
            Ok(marks(&out))
        }

        /// One read made while this thread holds the log's lock: `None`
        /// where the read waited for the lock.
        fn reads_beside_the_lock(
            &mut self,
            log: &Log,
        ) -> Option<Result<Vec<u32>, LogPositionLost>> {
            let (done, read) = mpsc::channel();
            thread::scope(|scope| {
                let held = log.held();
                scope.spawn(move || {
                    let _ = done.send(self.reads(log, 8));
                });
                let read = read.recv_timeout(Duration::from_secs(10)).ok();
                drop(held);
                read
            })
        }
    }

    #[test]
    fn a_reader_at_its_stamp_takes_no_lock_until_the_log_moves() {
        let log = Log::new(16);
        let mut reader = Reader::default();
        log.term_rose(2);
        member(&log, 1);
        assert_eq!(reader.reads(&log, 8), Ok(vec![TERM, MEMBER + 1]));
        // The read reached the log's end: the next reads nothing, and
        // waits for no lock to learn it.
        assert_eq!(reader.reads_beside_the_lock(&log), Some(Ok(vec![])));
        // A write moves the log past the stamp, and the read of it sets
        // the next.
        member(&log, 2);
        assert_eq!(reader.reads(&log, 8), Ok(vec![MEMBER + 2]));
        assert_eq!(reader.reads_beside_the_lock(&log), Some(Ok(vec![])));
        // So does a standby's mark.
        log.adopt(at(4, 3)).expect("the next position").mark();
        assert_eq!(reader.reads(&log, 8), Ok(vec![TERM]));
        assert_eq!(reader.reads_beside_the_lock(&log), Some(Ok(vec![])));
        // And a restart, which writes no entry: the reader stood
        // elsewhere, and hears it at each read from there.
        log.restart(at(11, 4)).expect("a place the log takes");
        assert_eq!(reader.reads(&log, 8), Err(LogPositionLost));
        assert_eq!(reader.reads(&log, 8), Err(LogPositionLost));
    }

    #[test]
    fn a_read_that_stops_short_of_the_logs_end_sets_no_stamp() {
        let log = Log::new(16);
        for epoch in 1..=5 {
            member(&log, epoch);
        }
        log.term_rose(2);
        member(&log, 6);
        // A full batch: the next reads what it left behind.
        let mut reader = Reader::default();
        assert_eq!(reader.reads(&log, 2), Ok(vec![MEMBER + 1, MEMBER + 2]));
        assert_eq!(reader.reads(&log, 2), Ok(vec![MEMBER + 3, MEMBER + 4]));
        // A batch that ends where the term changed, short of its bound.
        assert_eq!(reader.reads(&log, 8), Ok(vec![MEMBER + 5]));
        assert_eq!(reader.reads(&log, 8), Ok(vec![TERM, MEMBER + 6]));
        // The log's end: the stamp is set.
        assert_eq!(reader.reads_beside_the_lock(&log), Some(Ok(vec![])));
    }

    /// A search from the head answers, for every place of rings of every
    /// small size, full or not, as a count over the whole ring does: the
    /// events at or before the place, and with them whether the ring has the
    /// place under its term.
    #[test]
    fn a_search_from_the_head_answers_as_a_count_of_the_whole_ring() {
        for capacity in [1, 2, 3, 5, 16, 64] {
            for pushed in 0..=capacity + 20 {
                // Places three apart from 10: the ring's events leave
                // places between them.
                let mut kept = Kept::of(capacity);
                let mut next = 10;
                for _ in 0..pushed {
                    let entry = Entry {
                        at: at(next, 1),
                        event: Event::Term,
                    };
                    kept.push(capacity, entry);
                    next += 3;
                }
                for seq in 0..next + 2 {
                    let before = kept
                        .events
                        .iter()
                        .filter(|entry| entry.at.seq <= seq)
                        .count();
                    let of = format!("place {seq} of {pushed} pushed to {capacity}");
                    assert_eq!(kept.until(seq), before, "{of}");
                    let held = kept.events.iter().any(|entry| entry.at.seq == seq);
                    assert_eq!(
                        kept.has_at(at(seq, 1), before),
                        held || at(seq, 1) == kept.floor,
                        "{of}"
                    );
                    assert!(!kept.has_at(at(seq, 2), before), "{of}, under another term");
                }
            }
        }
    }

    /// The search takes its looks by a place's distance from the end, not
    /// by the row's length: one look where nothing follows the place, two
    /// where one event does (one in a row of one), and at most twice the
    /// bits its distance is written in. The bench holds the time of it,
    /// this case its shape.
    #[test]
    fn a_search_from_the_head_takes_its_looks_by_the_distance_from_it() {
        for len in [1usize, 2, 3, 7, 100, 4_096, 65_536] {
            let near = (0..=len).take(130);
            let far = (7..=16).flat_map(|bit| [(1usize << bit) - 1, 1 << bit, (1 << bit) + 1]);
            for behind in near.chain(far.filter(|&behind| behind <= len)) {
                let first = len - behind;
                let mut looks = 0u32;
                let found = first_after(len, |index| {
                    looks += 1;
                    index < first
                });
                let of = format!("{behind} behind the end of {len}");
                assert_eq!(found, first, "{of}");
                let bits = usize::BITS - behind.leading_zeros();
                assert!(looks <= (2 * bits).max(1), "{looks} looks at {of}");
                match behind {
                    0 => assert_eq!(looks, 1, "{of}"),
                    1 => assert_eq!(looks, 1 + u32::from(len > 1), "{of}"),
                    _ => {}
                }
            }
        }
    }

    /// A reader's place is searched once, and what the search hands it is
    /// what a count of the ring finds: for every place of a log, full or
    /// not, under its term and under two others, a reader is handed nothing
    /// unless the ring has the place and every event after it is retained,
    /// then the ring's events after the place. The searches are counted:
    /// one where the place is not behind the ring's floor, and none where
    /// it is. The rejoin's question answers as the hand does.
    #[test]
    fn a_reader_is_handed_what_a_count_of_the_ring_finds_after_its_place() {
        let has = |kept: &Kept, place: Position| {
            place == kept.floor || kept.events.iter().any(|entry| entry.at == place)
        };
        let after = |kept: &Kept, seq: u64| {
            kept.events
                .iter()
                .filter(|entry| entry.at.seq > seq)
                .count()
        };
        for watched in [1, 2, 5, 16] {
            for writes in 0..=2 * watched + 3 {
                let log = Log::new(watched);
                log.term_rose(1);
                for turn in 0..writes {
                    let mark = u32::try_from(turn).expect("a test's turn");
                    if turn % 3 == 0 {
                        member(&log, mark);
                    } else {
                        status(&log, mark);
                    }
                }
                let ring = log.ring.lock();
                for seq in 0..ring.head.seq + 2 {
                    for term in 0..3 {
                        let place = at(seq, term);
                        let retained = seq >= ring.watched.floor.seq;
                        let counted = (retained && has(&ring.watched, place))
                            .then(|| after(&ring.watched, seq));
                        let of = format!("{place:?} after {writes} writes to {watched}");
                        SEARCHES.with(|searches| searches.set(0));
                        let handed = ring.after(place).map(|events| events.len());
                        let searched = SEARCHES.with(std::cell::Cell::take);
                        assert_eq!(handed, counted, "{of}");
                        assert_eq!(
                            ring.continues(place),
                            counted.is_some(),
                            "the question, {of}"
                        );
                        assert_eq!(searched, u32::from(retained), "searches, {of}");
                    }
                }
            }
        }
    }

    #[test]
    fn an_applied_event_takes_the_leaders_position_and_term() {
        let leader = Log::new(32);
        status(&leader, 1);
        leader.term_rose(4);
        status(&leader, 2);
        leader.term_rose(6);
        member(&leader, 1);
        status(&leader, 3);
        let led = all(&leader);
        assert_eq!(
            places(&led),
            [at(1, 0), at(2, 4), at(3, 4), at(4, 6), at(5, 6), at(6, 6)]
        );
        // The follower led a term of its own once: an applied write never
        // takes it. It joined from a snapshot at the leader's first take.
        let follower = Log::new(32);
        follower.term_rose(3);
        follower.restart(at(2, 4)).expect("a place the log takes");
        applies(&follower, &led[2..]);
        assert_eq!(follower.head(), leader.head());
        assert_eq!(follower.floor(), at(2, 4));
        assert_eq!(all_after(&follower, at(2, 4)), led[2..]);
        for held in [at(3, 4), at(5, 6), at(6, 6)] {
            assert!(follower.holds(held), "{held:?}");
            assert!(leader.holds(held), "{held:?}");
        }
        assert!(
            !follower.holds(at(3, 3)),
            "never under the term the follower led"
        );
        assert_eq!(follower.term(), 6, "the highest term it followed");
    }

    #[test]
    fn an_applied_event_is_refused_at_a_gap_a_replay_or_a_falling_term() {
        let log = Log::new(16);
        assert_eq!(log.adopt(at(2, 3)).err(), Some(LogPositionLost), "a gap");
        assert_eq!(log.adopt(at(1, 0)).err(), Some(LogPositionLost), "no term");
        apply(&log, at(1, 3), 1);
        assert_eq!(log.head(), at(1, 3));
        assert_eq!(log.adopt(at(1, 3)).err(), Some(LogPositionLost), "a replay");
        assert_eq!(log.adopt(at(3, 3)).err(), Some(LogPositionLost), "a gap");
        assert_eq!(
            log.adopt(at(2, 2)).err(),
            Some(LogPositionLost),
            "a falling term"
        );
        assert!(log.adopt(at(2, 3)).is_ok());
        // A term below one the log has seen, wherever its head stands.
        let log = Log::new(16);
        log.term_rose(7);
        assert_eq!(
            log.adopt(at(2, 6)).err(),
            Some(LogPositionLost),
            "below a term the log took"
        );
        // A restart is refused as the event is, and leaves the log as it was.
        assert_eq!(
            log.restart(at(10, 6)),
            Err(LogPositionLost),
            "below a term the log took"
        );
        assert_eq!(
            log.restart(at(10, 7)),
            Err(LogPositionLost),
            "the term the store leads"
        );
        assert_eq!(log.restart(at(10, 0)), Err(LogPositionLost), "no term");
        assert_eq!((log.head(), log.followed()), (at(1, 7), None));
        assert!(log.adopt(at(2, 8)).is_ok());
    }

    #[test]
    fn an_own_write_under_no_term_takes_no_number_an_earlier_one_took() {
        let log = Log::new(16);
        let mut out = Vec::new();
        // A store that never restarted writes one number after another.
        member(&log, 1);
        assert_eq!(log.head(), at(1, 0));
        assert_eq!(log.read(at(1, 0), 8, &mut out), Ok(()));
        // Restored at a leader's place, it writes once more, and a reader
        // reads that write.
        log.restart(at(2, 2)).expect("a place the log takes");
        member(&log, 2);
        assert_eq!(log.read(at(2, 2), 8, &mut out), Ok(()));
        assert_eq!(places(&out), [at(3, 0)]);
        // Restored again at the same number: the log no longer has the
        // write that reader stands at, and the next one is numbered past
        // it, so the reader is never served another line as its own.
        log.restart(at(2, 3)).expect("a place the log takes");
        assert_eq!(log.read(at(3, 0), 8, &mut out), Err(LogPositionLost));
        member(&log, 3);
        assert_eq!(log.read(at(3, 0), 8, &mut out), Err(LogPositionLost));
        assert_eq!(log.head(), at(4, 0));
        // The restart's own place, a leader's, continues its reader.
        assert_eq!(log.read(at(2, 3), 8, &mut out), Ok(()));
        assert_eq!(places(&out), [at(4, 0)]);
        assert_eq!(marks(&out), [MEMBER + 3]);
    }

    #[test]
    fn a_reader_at_a_restarted_logs_own_write_reads_on() {
        let log = Log::new(16);
        let mut out = Vec::new();
        log.restart(at(2, 2)).expect("a place the log takes");
        member(&log, 1);
        // The head is the store's own write, where a cut of the store
        // stands: its reader continues from it, with nothing to read yet.
        let cut = log.head();
        assert_eq!(cut, at(3, 0));
        assert_eq!(log.read(cut, 8, &mut out), Ok(()));
        assert!(out.is_empty());
        member(&log, 2);
        status(&log, 9);
        assert_eq!(log.read(cut, 8, &mut out), Ok(()));
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
        /// A status write of the store's own.
        Row,
        /// A leader's status write, applied at this place.
        Led(u64, u64),
        /// The store takes this term.
        Take(u64),
    }

    fn make(log: &Log, moves: &[Move]) {
        for made in moves {
            match *made {
                Move::Restart(seq, term) => {
                    log.restart(at(seq, term)).expect("a place the log takes");
                }
                Move::Own => member(log, 1),
                Move::Row => status(log, 1),
                Move::Led(seq, term) => apply(log, at(seq, term), 1),
                Move::Take(term) => log.term_rose(term),
            }
        }
    }

    #[test]
    fn a_place_under_no_term_names_one_write_for_as_long_as_the_log_lives() {
        use Move::{Led, Own, Restart, Row, Take};
        // A reader stands at the last of the first moves, a write of the
        // store's own under no term. A restart follows: the reader is lost
        // at every read after it, whatever is written since, and a reader
        // that starts again at the head reads on.
        let walks: [(&str, &[Move], &[Move]); 9] = [
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
                "a status write",
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
        ];
        for (walk, first, then) in walks {
            let log = Log::new(16);
            let mut out = Vec::new();
            make(&log, first);
            let stood = log.head();
            assert_eq!(stood.term, 0, "{walk}");
            for (nth, made) in then.iter().enumerate() {
                make(&log, std::slice::from_ref(made));
                let read = log.read(stood, 8, &mut out);
                assert_eq!(read, Err(LogPositionLost), "{walk}: after move {nth}");
                let read = log.read(log.head(), 8, &mut out);
                assert_eq!(read, Ok(()), "{walk}: at the head after move {nth}");
            }
        }
    }

    #[test]
    fn a_leaders_positions_follow_its_head_whatever_its_own_line_reached() {
        let leader = Log::new(16);
        // Own writes under no term reach position 4. A restart sets the
        // head below them, and the store takes a term.
        leader.restart(at(2, 2)).expect("a place the log takes");
        member(&leader, 1);
        member(&leader, 2);
        assert_eq!(leader.head(), at(4, 0));
        leader.restart(at(1, 5)).expect("a place the log takes");
        leader.term_rose(6);
        status(&leader, 1);
        member(&leader, 3);
        // A term's positions are one after another from the head: what a
        // follower applies, and no gap is applied.
        let led = all_after(&leader, at(1, 5));
        assert_eq!(places(&led), [at(2, 6), at(3, 6), at(4, 6)]);
        let follower = Log::new(16);
        follower.restart(at(1, 5)).expect("a place the log takes");
        applies(&follower, &led);
        assert_eq!(follower.head(), leader.head());
        // Position 4 is the leader's now: the reader of the own write that
        // stood there is lost.
        let mut out = Vec::new();
        assert_eq!(leader.read(at(4, 0), 8, &mut out), Err(LogPositionLost));
    }

    #[test]
    fn a_place_no_write_took_is_not_kept_for_the_stores_next_write() {
        let log = Log::new(16);
        log.term_rose(2);
        let adopting = log.adopt(at(2, 9)).expect("the next position");
        assert!(!adopting.landed());
        drop(adopting);
        status(&log, 1);
        assert_eq!(
            log.head(),
            at(2, 2),
            "the store's own write, under its own term"
        );
    }

    #[test]
    fn a_guard_gives_up_its_own_place_and_no_later_guards() {
        let log = Log::new(16);
        // An applier that keeps one binding for its guards: the second
        // guard is made before the first drops.
        let mut adopting = log.adopt(at(1, 3)).expect("the next position");
        status_at(&log, at(1, 3), 1);
        assert!(adopting.landed());
        adopting = log.adopt(at(2, 3)).expect("the next position");
        status_at(&log, at(2, 3), 2);
        assert!(adopting.landed(), "the first guard's drop took the place");
        // An event tried again while its first guard stands: neither the
        // first guard's mark nor its drop takes the second's place.
        let first = log.adopt(at(3, 3)).expect("the next position");
        let second = log.adopt(at(3, 3)).expect("the same position");
        first.mark();
        assert_eq!(log.head(), at(2, 3), "the first guard's mark wrote");
        drop(first);
        status_at(&log, at(3, 3), 3);
        assert!(second.landed(), "the first guard's drop took the place");
        assert_eq!(marks(&all(&log)), [1, 2, 3]);
    }

    #[test]
    fn the_count_of_adoptions_wraps_and_a_guard_still_reads_its_own() {
        let log = Log::new(16);
        // One short of the count's last value: the first guard takes that
        // value and the second wraps to zero.
        log.ring.lock().adoptions = u32::MAX - 1;
        let first = log.adopt(at(1, 3)).expect("the next position");
        let second = log.adopt(at(1, 3)).expect("the same position");
        // The first guard's is not the last asked for, on either side of
        // the wrap: its drop leaves the second's place.
        drop(first);
        status_at(&log, at(1, 3), 1);
        assert!(second.landed(), "the first guard's drop took the place");
        // The last asked for gives its own place up.
        drop(log.adopt(at(2, 3)).expect("the next position"));
        status_at(&log, at(2, 3), 2);
        assert_ne!(log.head(), at(2, 3), "a dropped guard's place was taken");
    }

    /// The lock is the log's first field, by the order the log declares
    /// (`repr(C)`): a read addresses the ring off the log's own pointer.
    /// With the lock further in, a read keeps the lock's address in a
    /// register of its own, and its loop runs three instructions more an
    /// entry. Left to the compiler, the order follows the ring's size:
    /// eight bytes more moved the lock behind two fields.
    #[test]
    fn the_logs_lock_is_its_first_field() {
        assert_eq!(std::mem::offset_of!(Log, ring), 0);
    }

    #[test]
    fn a_write_of_the_stores_own_takes_no_place_a_guard_holds_and_ends_the_hold() {
        for own in ["a status write", "a member"] {
            for led in ["a status write", "a member", "a term's start"] {
                let log = Log::new(16);
                apply(&log, at(1, 3), 1);
                // The store writes of its own while a guard holds the next
                // place for its leader's event: a poll's answer that was
                // out when the store stopped leading.
                let adopting = log.adopt(at(2, 4)).expect("the next position");
                match own {
                    "a status write" => status(&log, 9),
                    _ => member(&log, 9),
                }
                // It stands where the store's own writes stand, under no
                // term, and the hold ended with it: the leader's event
                // lands nowhere, and the log follows no term by it.
                match led {
                    "a status write" => status_at(&log, at(2, 4), 2),
                    "a member" => member_at(&log, at(2, 4), 2),
                    _ => log.term_followed(at(2, 4)),
                }
                assert!(!adopting.landed(), "{own} inside {led}");
                assert_eq!(
                    places(&all(&log)),
                    [at(1, 3), at(2, 0)],
                    "{own} inside {led}"
                );
                assert_eq!(log.term(), 3, "{own} inside {led}");
                // Nothing of that leader's continues the log from there:
                // the standby starts again from a snapshot.
                drop(adopting);
                assert_eq!(
                    log.adopt(at(3, 4)).err(),
                    Some(LogPositionLost),
                    "{own} inside {led}"
                );
            }
        }
    }

    #[test]
    fn a_take_ends_an_application_and_starts_its_term_after_the_head() {
        let log = Log::new(16);
        apply(&log, at(1, 3), 1);
        let adopting = log.adopt(at(2, 3)).expect("the next position");
        // The store takes a lease while another leader's event is half
        // applied: the event's place is dropped, never taken by the mark.
        log.term_rose(4);
        assert!(!adopting.landed());
        assert_eq!(log.head(), at(2, 4));
        status(&log, 2);
        assert_eq!(places(&all(&log)), [at(1, 3), at(2, 4), at(3, 4)]);
        assert_eq!(marks(&all(&log)), [1, TERM, 2]);
    }

    #[test]
    fn an_appliers_write_that_a_take_overtook_stands_as_the_stores_own() {
        for led in ["a status write", "a member", "a term's start"] {
            let log = Log::new(16);
            apply(&log, at(1, 3), 1);
            // The store takes a lease between the applier's guard and its
            // write: the tables take the row under the store's own lead,
            // and no snapshot follows a take.
            let adopting = log.adopt(at(2, 3)).expect("the next position");
            log.term_rose(4);
            match led {
                "a status write" => status_at(&log, at(2, 3), 2),
                "a member" => member_at(&log, at(2, 3), 2),
                _ => log.term_followed(at(2, 3)),
            }
            // The event landed at no place of its leader's. A row stands
            // as a write of the store's own under the term taken, so the
            // log holds what the tables hold; a term's start wrote no row
            // and takes no place.
            assert!(!adopting.landed(), "{led}");
            let (places_held, marks_held) = match led {
                "a status write" => (vec![at(1, 3), at(2, 4), at(3, 4)], vec![1, TERM, 2]),
                "a member" => (
                    vec![at(1, 3), at(2, 4), at(3, 4)],
                    vec![1, TERM, MEMBER + 2],
                ),
                _ => (vec![at(1, 3), at(2, 4)], vec![1, TERM]),
            };
            assert_eq!(places(&all(&log)), places_held, "{led}");
            assert_eq!(marks(&all(&log)), marks_held, "{led}");
            assert_eq!(log.leads(4), Ok(()), "{led}");
        }
        // A store that leads an earlier term than the event's took nothing
        // since the guard: a late write of its own ends the hold, and the
        // applier's write is let go, as on a store that follows.
        let log = Log::new(16);
        log.term_rose(5);
        let adopting = log.adopt(at(2, 6)).expect("the next position");
        status(&log, 9);
        status_at(&log, at(2, 6), 2);
        assert!(!adopting.landed());
        assert_eq!(places(&all(&log)), [at(1, 5), at(2, 5)]);
        assert_eq!(marks(&all(&log)), [TERM, 9]);
        // A store that follows took no term either. A later guard's write
        // landed under a later term: the earlier guard's is let go.
        let log = Log::new(16);
        apply(&log, at(1, 3), 1);
        let earlier = log.adopt(at(2, 3)).expect("the next position");
        let later = log.adopt(at(2, 4)).expect("the next position");
        member_at(&log, at(2, 4), 2);
        assert!(later.landed());
        member_at(&log, at(2, 3), 9);
        assert!(!earlier.landed());
        assert_eq!(places(&all(&log)), [at(1, 3), at(2, 4)]);
        assert_eq!(marks(&all(&log)), [1, MEMBER + 2]);
    }

    #[test]
    fn a_store_leads_the_term_its_take_started_and_the_log_says_why_it_leads_no_other() {
        let log = Log::new(16);
        assert_eq!(log.leads(3), Err(NotLeading::NotYet), "before a take");
        log.term_rose(3);
        assert_eq!(log.leads(3), Ok(()));
        assert_eq!(
            log.leads(4),
            Err(NotLeading::NotYet),
            "a term the log has not seen"
        );
        assert_eq!(
            log.leads(2),
            Err(NotLeading::Past(3)),
            "a term below its own"
        );
        // The store follows term 5: it leads neither that term nor its old one.
        apply(&log, at(2, 5), 1);
        assert_eq!(log.leads(3), Err(NotLeading::Follows(5)), "the term it led");
        assert_eq!(
            log.leads(5),
            Err(NotLeading::Follows(5)),
            "the term it follows"
        );
        // A take of the very term it followed starts no lead. A later one does.
        log.term_rose(5);
        assert_eq!(
            log.leads(5),
            Err(NotLeading::Follows(5)),
            "a take that started no term"
        );
        log.term_rose(6);
        assert_eq!(log.leads(6), Ok(()));
    }

    #[test]
    fn a_store_leads_from_a_take_that_starts_a_term_until_it_follows() {
        let log = Log::new(16);
        assert_eq!(log.followed(), Some(0), "before a take");
        log.term_rose(3);
        assert_eq!(log.followed(), None, "its take");
        // Each way of following ends the lead: an applied event, a term's
        // start marked, a restart at a snapshot.
        apply(&log, at(2, 5), 1);
        assert_eq!(log.followed(), Some(5), "an applied event");
        log.term_rose(6);
        assert_eq!(log.followed(), None, "a later take");
        log.adopt(at(4, 7)).expect("the next position").mark();
        assert_eq!(log.followed(), Some(7), "a term's start marked");
        log.term_rose(8);
        assert_eq!(log.followed(), None, "a later take");
        log.restart(at(40, 9)).expect("a place the log takes");
        assert_eq!(log.followed(), Some(9), "a restart");
        // A take at or below the term the log followed starts no lead: it
        // writes no position, and the store's next write is under no term.
        for take in [9, 4] {
            log.term_rose(take);
            assert_eq!(log.followed(), Some(9), "a take of term {take}");
            assert_eq!(log.head(), at(40, 9));
        }
        status(&log, 2);
        assert_eq!(log.head(), at(41, 0));
        log.term_rose(10);
        assert_eq!(log.followed(), None, "a take above every term seen");
        assert_eq!(log.head(), at(42, 10));
    }

    #[test]
    fn a_restart_under_an_open_guard_leaves_no_adopted_place() {
        let log = Log::new(16);
        apply(&log, at(1, 3), 1);
        // A snapshot is restored while a leader's event is half applied.
        let adopting = log.adopt(at(2, 3)).expect("the next position");
        log.restart(at(40, 5)).expect("a place the log takes");
        // The store's next write follows the snapshot's place: the place of
        // the event that never landed went with its history.
        status(&log, 2);
        assert_eq!(log.head(), at(41, 0));
        assert!(!adopting.landed());
        // The guard that outlived its history marks nothing.
        adopting.mark();
        drop(adopting);
        assert_eq!(log.head(), at(41, 0));
        assert_eq!(log.adopt(at(42, 5)).err(), Some(LogPositionLost));
    }

    #[test]
    fn a_log_restarted_at_a_snapshot_has_that_place_alone_and_continues_after_it() {
        let log = Log::new(16);
        log.term_rose(4);
        for mark in 0..3 {
            status(&log, mark);
        }
        log.restart(at(100, 7)).expect("a place the log takes");
        assert_eq!(log.term(), 7, "the term of the snapshot's place");
        assert_eq!(log.head(), at(100, 7));
        assert_eq!(log.floor(), at(100, 7));
        assert!(
            log.holds(at(100, 7)),
            "its place, though no event is retained"
        );
        assert!(!log.holds(at(100, 4)));
        assert!(
            !log.holds(at(3, 4)),
            "what stood before the restart is gone"
        );
        assert!(!log.holds(at(0, 0)));
        let mut out = Vec::new();
        assert_eq!(log.read(at(100, 7), 8, &mut out), Ok(()));
        assert!(out.is_empty());
        assert_eq!(log.read(at(99, 7), 8, &mut out), Err(LogPositionLost));
        assert_eq!(
            log.read(at(4, 4), 8, &mut out),
            Err(LogPositionLost),
            "a reader at the head before the restart"
        );
        assert_eq!(
            log.read(at(100, 4), 8, &mut out),
            Err(LogPositionLost),
            "the same position of another history"
        );
        apply(&log, at(101, 7), 9);
        assert_eq!(log.read(at(100, 7), 8, &mut out), Ok(()));
        assert_eq!(marks(&out), [9]);
        assert_eq!(out.first().map(|entry| entry.at), Some(at(101, 7)));
    }

    #[test]
    fn a_log_says_its_first_place_while_it_is_its_stores_whole_history() {
        // Four writes are retained, and the places start after a number
        // the run drew.
        assert_eq!(Log::new(4).first(), Some(at(0, 0)), "no draw");
        let log = Log::new(4).starting_after(1_000);
        assert_eq!(log.head(), at(1_000, 0));
        assert_eq!(log.floor(), at(1_000, 0));
        assert_eq!(log.first(), Some(at(1_000, 0)));
        assert!(!log.holds(at(0, 0)), "no place of this run");
        log.term_rose(3);
        status(&log, 1);
        status(&log, 2);
        assert_eq!(
            places(&all_after(&log, at(1_000, 0))),
            [at(1_001, 3), at(1_002, 3), at(1_003, 3)]
        );
        status(&log, 3);
        assert_eq!(log.first(), Some(at(1_000, 0)), "every write retained");
        // The ring lets its first write go: a reader that holds nothing
        // starts from a snapshot.
        member(&log, 1);
        assert_eq!(log.first(), None, "the term's start is let go");
    }

    #[test]
    fn a_log_restarted_at_a_snapshots_place_has_no_first_place() {
        let log = Log::new(16).starting_after(1_000);
        log.restart(at(1_050, 7)).expect("a place the log takes");
        // What the store held at the snapshot is no event of the log.
        assert_eq!(log.first(), None);
        // The first place's number stays one of the store's own line: after
        // a restart below it, a write under no term is numbered past it.
        let log = Log::new(16).starting_after(1_000);
        log.restart(at(5, 7)).expect("a place the log takes");
        member(&log, 1);
        assert_eq!(log.head(), at(1_001, 0));
        assert_eq!(log.first(), None);
    }

    #[test]
    fn a_log_that_holds_nothing_joins_its_leaders_history_at_the_first_position() {
        // The leader's run drew below this run's number, above it, and the
        // same one.
        for (led_after, own) in [(1_000, 2_000), (2_000, 1_000), (1_000, 1_000)] {
            let leader = Log::new(32).starting_after(led_after);
            leader.term_rose(4);
            member(&leader, 1);
            status(&leader, 2);
            let first = at(led_after, 0);
            let led = all_after(&leader, first);
            assert_eq!(marks(&led), [TERM, MEMBER + 1, 2]);
            let follower = Log::new(32).starting_after(own);
            // A drawn place is no start a guard takes, whatever comes next.
            assert_eq!(
                follower.adopt(led[0].at).err(),
                Some(LogPositionLost),
                "{own}"
            );
            assert_eq!(follower.join(led[0].at), Ok(()), "{own}");
            assert_eq!((follower.term(), follower.followed()), (4, Some(4)));
            applies(&follower, &led[1..]);
            assert_eq!(follower.head(), leader.head());
            // Its log is its leader's from the first place on: a reader
            // that holds nothing joins it there.
            assert_eq!(follower.first(), Some(first), "{own}");
            assert_eq!(all_after(&follower, first), led);
            // A write of its own under no term follows the head, past the
            // number its own run drew: that number stays on its own line.
            status(&follower, 9);
            let past = (led_after + 3).max(own) + 1;
            assert_eq!(follower.head(), at(past, 0), "{own}");
        }
    }

    #[test]
    fn a_join_is_refused_once_the_log_holds_a_history_and_under_no_term() {
        let log = Log::new(16).starting_after(1_000);
        assert_eq!(log.join(at(501, 0)), Err(LogPositionLost), "no term");
        assert_eq!(
            log.join(at(0, 3)),
            Err(LogPositionLost),
            "no place before it"
        );
        assert_eq!(
            (log.head(), log.first(), log.term()),
            (at(1_000, 0), Some(at(1_000, 0)), 0),
            "the log as it was"
        );
        // A write of its own, a take, a snapshot's place, a join: each is
        // a history the log holds, and no other starts in it.
        for held in ["a write", "a take", "a restart", "a join"] {
            let log = Log::new(16).starting_after(1_000);
            match held {
                "a write" => status(&log, 1),
                "a take" => log.term_rose(2),
                "a restart" => log.restart(at(40, 3)).expect("a place the log takes"),
                _ => log.join(at(501, 3)).expect("a log that holds nothing"),
            }
            let (head, first) = (log.head(), log.first());
            assert_eq!(log.join(at(601, 5)), Err(LogPositionLost), "{held}");
            assert_eq!((log.head(), log.first()), (head, first), "{held}");
        }
    }

    #[test]
    fn a_joined_log_follows_its_leaders_term_and_a_take_of_it_again_starts_no_lead() {
        // The leader's run drew 1,000 and led term 1; this run drew 500.
        let told = Arc::new(AtomicUsize::new(0));
        let log = {
            let told = Arc::clone(&told);
            Log::new(16).starting_after(500).on_watched(move || {
                told.fetch_add(1, Ordering::Relaxed);
            })
        };
        log.join(at(1_001, 1)).expect("a log that holds nothing");
        assert_eq!((log.head(), log.followed()), (at(1_001, 1), Some(1)));
        // The listener is told at the join's end, as at a mark's.
        assert_eq!(told.load(Ordering::Relaxed), 1);
        // A shared lease whose record was lost counts its terms again. The
        // take of term 1 leads nothing here: the places under that term
        // stay the leader's writes alone.
        log.term_rose(1);
        status(&log, 1);
        assert_eq!((log.head(), log.followed()), (at(1_002, 0), Some(1)));
        assert!(!log.holds(at(1_002, 1)));
        // The next term is led from the head.
        log.term_rose(2);
        assert_eq!((log.head(), log.followed()), (at(1_003, 2), None));
    }

    #[test]
    fn no_join_leaves_a_log_empty_at_another_place() {
        // Every way through the door: the log is as it was, or it holds
        // the term's start it joined at and has seen that term.
        for joined in [
            at(501, 3),
            at(1_001, 3),
            at(2_001, 1),
            at(1, 1),
            at(501, 0),
            at(0, 3),
        ] {
            let log = Log::new(16).starting_after(1_000);
            let before = (log.head(), log.first(), log.term());
            match log.join(joined) {
                Ok(()) => {
                    let first = at(joined.seq - 1, 0);
                    assert_eq!((log.head(), log.first()), (joined, Some(first)));
                    assert_eq!(all_after(&log, first).len(), 1, "{joined:?}");
                    assert_eq!(log.followed(), Some(joined.term), "{joined:?}");
                }
                Err(LogPositionLost) => {
                    assert_eq!(joined.seq.min(joined.term), 0, "{joined:?}");
                    assert_eq!((log.head(), log.first(), log.term()), before);
                }
            }
        }
    }

    #[test]
    fn a_join_under_an_open_guard_leaves_no_adopted_place() {
        // A log that counts from zero gives a guard at its start.
        let log = Log::new(16);
        let adopting = log.adopt(at(1, 3)).expect("the next position");
        log.join(at(7, 3)).expect("a log that holds nothing");
        // The store's next write follows the joined place: the place of
        // the event that never landed went with the start it stood after.
        status(&log, 2);
        assert_eq!(log.head(), at(8, 0));
        assert!(!adopting.landed());
        adopting.mark();
        drop(adopting);
        assert_eq!(log.head(), at(8, 0));
    }

    #[test]
    fn a_joined_logs_own_line_stands_past_its_first_place_and_the_number_it_drew() {
        // The leader's run drew 1,000; this run drew below it, and above.
        for (own, past) in [(500, 1_001), (2_000, 2_001)] {
            let log = Log::new(16).starting_after(own);
            log.join(at(1_001, 3)).expect("a log that holds nothing");
            // A snapshot's place below the first place, then one write of
            // the store's own under no term: it is numbered past the first
            // place and past the drawn number, so neither names it.
            log.restart(at(999, 4)).expect("a place the log takes");
            member(&log, 1);
            assert_eq!(log.head(), at(past, 0), "{own}");
            // A reader that joined at the first place is never served the
            // store's own line as its leader's: it starts again.
            let mut out = Vec::new();
            for held in [at(1_000, 0), at(own, 0)] {
                assert_eq!(log.read(held, 8, &mut out), Err(LogPositionLost), "{own}");
            }
        }
    }

    #[test]
    fn a_store_that_follows_writes_under_no_term_until_it_takes_one() {
        // The store leads term 3. A leader's event of term 5 lands by the
        // store's own write or by the mark, term 5 is a snapshot's, or the
        // store is told term 5 under the guard of its first position.
        for way in ["a write", "a mark", "a restart", "the term told"] {
            let log = Log::new(16);
            log.term_rose(3);
            status(&log, 1);
            assert_eq!(log.head(), at(2, 3), "its own write while it leads");
            match way {
                "a write" => apply(&log, at(3, 5), 7),
                "a mark" => log.adopt(at(3, 5)).expect("the next position").mark(),
                "a restart" => log.restart(at(3, 5)).expect("a place the log takes"),
                _ => {
                    let adopting = log.adopt(at(3, 5)).expect("the next position");
                    log.term_followed(at(3, 5));
                    assert!(adopting.landed());
                }
            }
            assert_eq!((log.head(), log.term()), (at(3, 5), 5), "{way}");
            // A late write of its own: the term it led is over.
            status(&log, 2);
            assert_eq!(log.head(), at(4, 0), "{way}");
            assert!(!log.holds(at(4, 0)), "{way}");
            assert!(!log.holds(at(4, 3)), "{way}");
            // No leader's history continues from it.
            assert_eq!(log.adopt(at(5, 5)).err(), Some(LogPositionLost), "{way}");
            // A take is above every term the log followed.
            log.term_rose(5);
            log.term_rose(4);
            assert_eq!(log.head(), at(4, 0), "{way}: no take at term 5");
            log.term_rose(6);
            status(&log, 3);
            assert_eq!(log.head(), at(6, 6), "{way}: it leads again");
        }
    }

    #[test]
    fn a_store_that_leads_applies_no_event_under_its_own_term() {
        let log = Log::new(16);
        log.term_rose(3);
        status(&log, 1);
        // Term 3 is this store's: an event under it is no leader's.
        assert_eq!(log.adopt(at(3, 3)).err(), Some(LogPositionLost));
        // A later term's event lands, and the store follows that term on.
        apply(&log, at(3, 4), 7);
        apply(&log, at(4, 4), 8);
        assert_eq!(log.head(), at(4, 4));
    }

    #[test]
    fn a_restored_leaders_late_write_takes_no_place_its_term_wrote_before() {
        // Three replicas. The first leads term 1 through position 4. One
        // follower applied all four; the other applied two and took term 2
        // at position 3: the shorter history.
        let (first, ahead, behind) = (Log::new(16), Log::new(16), Log::new(16));
        first.term_rose(1);
        for mark in 2..=4 {
            status(&first, mark);
        }
        let led = all(&first);
        applies(&ahead, &led);
        applies(&behind, &led[..2]);
        behind.term_rose(2);
        assert_eq!(behind.head(), at(3, 2));
        // The first, deposed, is not held there. It is restored from that
        // leader's snapshot, and one write of its own lands a moment later.
        assert!(!behind.holds(first.head()));
        first.restart(behind.head()).expect("a place the log takes");
        assert!(!first.holds(at(4, 1)), "a place of the history it left");
        status(&first, 99);
        assert_eq!(
            first.head(),
            at(4, 0),
            "term 1 wrote position 4 before, in another history"
        );
        // The replica that is ahead leads next and never followed term 2:
        // position 4 under term 1 is its own history's.
        ahead.term_rose(3);
        assert!(ahead.holds(at(4, 1)));
        assert!(!ahead.holds(first.head()));
        assert_eq!(
            first.adopt(at(5, 3)).err(),
            Some(LogPositionLost),
            "it starts again from a snapshot"
        );
    }

    #[test]
    fn a_term_told_while_its_start_is_applied_is_the_leaders_and_no_take() {
        let log = Log::new(16);
        apply(&log, at(1, 3), 1);
        // The standby's store follows the leader's term by its own entry,
        // at the place the guard of that term's first position holds.
        let adopting = log.adopt(at(2, 5)).expect("the next position");
        log.term_followed(at(2, 5));
        assert!(adopting.landed());
        drop(adopting);
        assert_eq!(places(&all(&log)), [at(1, 3), at(2, 5)]);
        assert_eq!(marks(&all(&log)), [1, TERM]);
        status(&log, 2);
        assert_eq!(
            log.head(),
            at(3, 0),
            "the store follows term 5 and does not lead it"
        );
    }
}
