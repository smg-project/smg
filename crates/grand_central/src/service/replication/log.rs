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
#[non_exhaustive]
pub enum Event {
    /// A membership or status write.
    Member {
        /// The membership epoch the write made.
        epoch: u64,
        change: MembershipChange,
        /// The row the change is about, as the write leaves it; `None` for
        /// a removal.
        row: Option<Arc<WorkerRow>>,
        /// The row the write took out: a removal's row, a replacement's old
        /// incarnation.
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
#[non_exhaustive]
pub struct Entry {
    pub at: Position,
    pub event: Event,
}

/// A reader's place is no longer the log's, or never was: it starts again
/// from a snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[expect(clippy::exhaustive_structs, reason = "no field: its name is its value")]
pub struct LogPositionLost;

/// Why a store does not lead a term its replica took ([`Log::leads`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
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
    /// carries no load row.
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
    /// log again for its words. Four callers read it, each in its own
    /// turn: the service's guard, the replication's check, the door's check
    /// of its term and the door's word for a status it writes. The check
    /// runs ahead of a life's mint, of each row a registration writes or
    /// removes and of a refresh's write, and once more behind a drain that
    /// wrote nothing. The word is asked under an object's lock and under
    /// the store's membership lock: this read takes the log's lock alone,
    /// and no holder of the log's lock takes one of those two. A write
    /// that names its term meets the same test once more, in the take of
    /// the log's lock that gives its event a place ([`Journal::member`]).
    pub fn leads(&self, term: u64) -> Result<(), NotLeading> {
        let ring = self.ring.lock();
        self.leads_in(&ring, term)
    }

    /// [`Log::leads`] under a hold of the log's lock.
    fn leads_in(&self, ring: &Ring, term: u64) -> Result<(), NotLeading> {
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
    ///
    /// `under` is the term the write is made under, where its writer names
    /// one. The log asks it here, in the take of its lock that gives the
    /// event its place: no term ends between this ask and the stamp. Where
    /// the store does not lead that term, nothing is appended, no
    /// applier's hold ends, and the answer is no. Every other write
    /// answers yes, one that is let go too: its store moves its tables as
    /// it did.
    fn push(&self, by: Option<Position>, under: Option<u64>, event: Event) -> bool {
        let mut ring = self.ring.lock();
        if under.is_some_and(|term| self.leads_in(&ring, term).is_err()) {
            return false;
        }
        let held = ring.adopting.take();
        let lands = by.filter(|at| held == Some(*at));
        // The hold ended before the applier's write: it is let go, or it
        // stands as the store's own.
        if let (Some(at), None) = (by, lands) {
            if !self.took_since(&ring, at) || matches!(event, Event::Term) {
                return true;
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
                        return true;
                    }
                    // Its number is the write's whole name: past the head,
                    // and past every own write a restart left above it.
                    seq = ring.fresh(seq);
                }
                Position { seq, term }
            }
        };
        self.append(&mut ring, at, event);
        true
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
    /// store keeps no load row. The listener is told at its end, under no
    /// lock of the log.
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
        term: Option<u64>,
        epoch: u64,
        change: MembershipChange,
        row: Option<&WorkerRow>,
        left: Option<&WorkerRow>,
    ) -> bool {
        // The rows are copied before the log's lock is taken.
        let copy = |row: &WorkerRow| Arc::new(row.clone());
        self.push(
            at,
            term,
            Event::Member {
                epoch,
                change,
                row: row.map(copy),
                left: left.map(copy),
            },
        )
    }

    fn generation(&self, at: Option<Position>, generation: u64) {
        // A generation names no term: it is never refused.
        self.push(at, None, Event::Generation { generation });
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
        self.push(Some(at), None, Event::Term);
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
#[path = "../../../tests/unit/service/replication/log/tests.rs"]
mod tests;
