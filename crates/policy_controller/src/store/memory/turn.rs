//! The files of leases and their turns: a file a decider, each behind a
//! turn of its own. Who holds a file's turn writes the file, and what a
//! session's frame says of a lease is handed to the lease's file, never
//! written by the session. A `Decide` stream hands the same way, the
//! release of a decision it stops at its send: what is said of a session
//! below holds for every hand.
//!
//! Which file. A decider issues into its own file, and a lease's id names
//! that file for as long as the lease can be on file
//! ([`MemoryStore::file_of`], by the blocks of the term's counter): a
//! frame finds its lease's file by its id, with no turn held. A session
//! reads its frame whole and hands what it says to that file's ring
//! ([`MemoryStore::hand`]). Whoever applies at a file applies everything
//! handed to it before it reads a record, ends a dispatch deadline or
//! issues: the record's read, the door's change and the word to the
//! engine, in the order the frames were handed, so one lease's events are
//! told in their order whoever applies them. A ring is written in place
//! and holds [`MemoryStore::HANDOVER`] frames. A hand that finds it full
//! stops until what it holds was applied, by the hand itself where it may
//! apply and by the file's decider otherwise: nothing handed is dropped
//! and no ring grows.
//!
//! Who applies. A decider with a job at hand is at work at its file
//! ([`MemoryStore::at_work`]): it says so, then takes its turn as the
//! first step of its work, tries the turn once before each decision reads
//! its rows' counts, takes it at each issue, takes it behind the give-up
//! of an answer its stream refused, and takes it once more before it
//! waits for its next job. While it is at work no other thread
//! applies at its file: a thread that holds the file's turn and finds its
//! decider at work lets go and applies nothing, and what was handed waits
//! for the decider's next take. So a file's lines stay with the one
//! thread that reads them. With the file's decider waiting for a job,
//! whoever takes the turn applies: the session that hands, and the
//! gateways' sweep, which ends the file's leases past their dispatch
//! deadline ([`LeasesStore::expire`](crate::store::LeasesStore::expire)).
//! So no handed frame waits on a lull, but behind a thread that left its
//! work by an unwind: what it left is the next take's.
//!
//! A decision reads its own file behind everything handed to it. A row
//! that another file's leases book lags by that file's next take: a
//! worker's count is one atomic a row, stepped under the turn of the file
//! that books it.
//!
//! The lock order, written here once: a file's turn is taken first or not
//! at all, and one turn at a time. The term's rise holds every turn, in
//! the files' order (`set_term`, `follow_term`), and so does a standby's
//! restore, from its start until it has read the ids on file
//! ([`MemoryStore::restore_from`]). Under a turn come the world's load
//! (no lock), a stage's own lock, the engine's sessions table, the store's
//! pins and the journal's word of a term. A standby's restore takes the
//! store's tables under its turns too: the workers', the health's and the
//! loads', in that order. No path takes one of those and then a turn. A
//! lifecycle call of a served controller therefore runs under a turn, or
//! after it where a decision gives its own lease up: it is bounded and
//! calls no door of the leases. A debug build marks the thread that holds
//! a turn and refuses a second take of it by that thread.

use std::{
    fmt,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{
            fence, AtomicU64, AtomicUsize,
            Ordering::{Relaxed, SeqCst},
        },
        mpsc::{sync_channel, Receiver, SyncSender, TrySendError},
        Arc, OnceLock,
    },
    thread,
    time::Duration,
};

use arc_swap::Guard;
use parking_lot::{Mutex, MutexGuard};

use super::{Key, Leases, MemoryStore};
use crate::{
    decider::Decider,
    lease::LeaseId,
    store::{
        leases::{LeaseRecord, LeaseState, Said},
        world::World,
        GatewayId,
    },
};

/// What tells the engine of a frame applied to a lease on file: the world
/// the frame is told under, the record as it stood before the frame, what
/// the frame said, and the decider whose file the lease is on, where the
/// store's host named its deciders.
type Teller = fn(&MemoryStore, &World, &LeaseRecord, Said, Option<Decider>);

/// The world a take's frames are told under: the store's, loaded at the
/// take's first tell. So a take that tells nothing loads no world, and one
/// that tells many loads one: each frame of a take is told under the
/// generation that stood at the take's first tell.
#[derive(Default)]
pub(super) struct Told(Option<Guard<Arc<World>>>);

impl Told {
    /// The world this take tells under.
    pub(super) fn world(&mut self, store: &MemoryStore) -> &World {
        self.0.get_or_insert_with(|| store.world.load())
    }
}

/// What a frame said of a gateway's lease, as it waits in the ring.
type Handed = (GatewayId, LeaseId, Said);

/// What a hand reads of a file, on lines of its own: the threads that
/// write the file share no line with the threads that only hand to it.
#[repr(align(128))]
struct Shown {
    /// The mark of the thread at work at the file ([`me`]); 0 for none.
    /// One thread at most works at a file: its decider, with a job at
    /// hand. It applies what is handed while it works, and once more as
    /// it leaves.
    at_work: AtomicUsize,
    /// The ring's writing end, every hand's.
    hands: SyncSender<Handed>,
}

/// What stands behind the turn: the file, and the reading end of its
/// ring.
pub(super) struct Behind {
    leases: Leases,
    ring: Receiver<Handed>,
}

/// One decider's file of leases behind its turn, with the ring of what is
/// handed to it.
pub(super) struct File {
    /// The turn: its holder writes the file.
    turn: Mutex<Behind>,
    /// What a hand reads.
    shown: Shown,
    /// The decider whose file this is, where the store's host named its
    /// deciders: a lease's events are told under it.
    name: Option<Decider>,
    /// The thread that holds the turn; 0 for none.
    #[cfg(debug_assertions)]
    holder: AtomicUsize,
}

impl File {
    fn new(leases: Leases, name: Option<Decider>) -> Self {
        let (hands, ring) = sync_channel(MemoryStore::HANDOVER);
        Self {
            turn: Mutex::new(Behind { leases, ring }),
            shown: Shown {
                at_work: AtomicUsize::new(0),
                hands,
            },
            name,
            #[cfg(debug_assertions)]
            holder: AtomicUsize::new(0),
        }
    }

    /// Takes the turn and applies nothing: for the term's read, which no
    /// handed frame changes.
    pub(super) fn take(&self) -> Turn<'_> {
        #[cfg(debug_assertions)]
        assert_ne!(
            self.holder.load(SeqCst),
            me(),
            "the turn of the leases' file is taken first or not at all: its holder took it again"
        );
        let behind = self.turn.lock();
        #[cfg(debug_assertions)]
        self.holder.store(me(), SeqCst);
        Turn {
            behind,
            #[cfg(debug_assertions)]
            file: self,
        }
    }

    /// Whether the turn is free at this instant, tried.
    #[cfg(test)]
    pub(super) fn free(&self) -> bool {
        self.turn.try_lock().is_some()
    }

    /// The turn where it is free, with nothing applied. `None` where a
    /// thread holds it: what is handed behind that holder's take waits
    /// for the next take that applies.
    pub(super) fn try_take(&self) -> Option<Turn<'_>> {
        let behind = self.turn.try_lock()?;
        #[cfg(debug_assertions)]
        self.holder.store(me(), SeqCst);
        Some(Turn {
            behind,
            #[cfg(debug_assertions)]
            file: self,
        })
    }
}

/// This thread's mark: the address of a cell of its own, never 0.
fn me() -> usize {
    thread_local!(static CELL: u8 = const { 0 });
    CELL.with(|cell| std::ptr::from_ref(cell).addr())
}

/// The turn, held.
pub(super) struct Turn<'a> {
    behind: MutexGuard<'a, Behind>,
    #[cfg(debug_assertions)]
    file: &'a File,
}

#[cfg(debug_assertions)]
impl Drop for Turn<'_> {
    fn drop(&mut self) {
        self.file.holder.store(0, SeqCst);
    }
}

impl Deref for Turn<'_> {
    type Target = Leases;

    fn deref(&self) -> &Leases {
        &self.behind.leases
    }
}

impl DerefMut for Turn<'_> {
    fn deref_mut(&mut self) -> &mut Leases {
        &mut self.behind.leases
    }
}

/// A decider's seat at its file ([`MemoryStore::at_work`]). While its
/// thread works the hands leave what they hand to that file to it: it
/// applies where it finds its turn free before a read of the counts
/// ([`AtWork::apply`]), at each issue, and behind the give-up of an
/// answer its stream refused. A thread that rests ([`AtWork::rest`]) or
/// drops its seat takes the turn once more: what was handed behind its
/// last take is applied before the thread waits.
pub struct AtWork {
    store: Arc<MemoryStore>,
    file: usize,
    working: bool,
}

impl AtWork {
    /// Applies what is handed, before a decision reads its rows' counts,
    /// where the turn is free. Where it is held the decision reads on:
    /// what is handed behind the holder's take waits for the next take
    /// that applies, and the decision's issue is one.
    pub fn apply(&self) {
        if let Some(mut turn) = self.store.files[self.file].try_take() {
            self.store
                .applied(self.file, &mut turn, &mut Told::default());
        }
    }

    /// The thread has a job at hand again: it says so and then takes its
    /// turn, the first step of its work.
    pub fn work(&mut self) {
        if !self.working {
            self.working = true;
            self.store.begin(self.file);
        }
    }

    /// The thread waits for its next job: it says so, and what was handed
    /// behind its last take is applied before it waits.
    pub fn rest(&mut self) {
        if self.working {
            self.working = false;
            self.store.leave(self.file, true);
        }
    }
}

impl Drop for AtWork {
    /// A thread that leaves by an unwind looks at nothing: what it left
    /// is the next take's, and a tell that unwound is not asked again
    /// under it.
    fn drop(&mut self) {
        if self.working {
            self.store.leave(self.file, !thread::panicking());
        }
    }
}

impl fmt::Debug for AtWork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AtWork").finish_non_exhaustive()
    }
}

/// A stream's hand at the leases' files: what its frames say of its
/// gateway's leases goes through it ([`MemoryStore::hand`]), each frame to
/// the file its lease's id names. It holds no file's ring: the store it
/// is handed through is the one whose files take its frames.
#[derive(Debug)]
pub struct Hand {
    gateway: GatewayId,
    /// The files the hand handed to, a bit a file: its stream's end looks
    /// at each once more ([`MemoryStore::hand_back`]).
    handed: [AtomicU64; Decider::COUNT / 64],
}

impl Hand {
    /// Opens a stream's hand for `gateway`'s frames.
    #[must_use]
    pub fn new(gateway: GatewayId) -> Self {
        Self {
            gateway,
            handed: Default::default(),
        }
    }

    /// The gateway whose leases the hand's frames name.
    pub fn gateway(&self) -> GatewayId {
        self.gateway
    }

    /// Notes that the hand handed to the file at `at`.
    fn mark(&self, at: usize) {
        let (word, bit) = (&self.handed[at / 64], 1 << (at % 64));
        if word.load(Relaxed) & bit == 0 {
            word.fetch_or(bit, Relaxed);
        }
    }
}

/// The looks a hand at a full ring yields between before it sleeps
/// between them ([`MemoryStore::hand`]).
const FULL_RING_YIELDS: u32 = 64;

/// How long a hand at a full ring sleeps between its looks once it has
/// yielded [`FULL_RING_YIELDS`] times: well under a decision's budget.
const FULL_RING_SLEEP: Duration = Duration::from_micros(50);

/// The bit of the files' fault count that says the last lifecycle call
/// panicked: a streak of faults writes one line.
const STREAK: u64 = 1 << (u64::BITS - 1);

/// The store's files of leases, one a decider, and the word their events
/// are told through.
pub(super) struct Files {
    files: Box<[File]>,
    /// The controller's word to the engine, given once.
    teller: OnceLock<Teller>,
    /// The lifecycle calls that panicked, below a bit that says the last
    /// call did ([`MemoryStore::lifecycle_faults`]): read beside the word
    /// at each tell, written at a fault and at the first return behind one.
    faults: AtomicU64,
}

impl Files {
    /// One file that no decider is named for: the store of a host that
    /// names none.
    pub(super) fn one(leases: Leases) -> Self {
        Self {
            files: Box::new([File::new(leases, None)]),
            teller: OnceLock::new(),
            faults: AtomicU64::new(0),
        }
    }

    /// A file a decider, each over the leases `leases` makes: a host
    /// names [`Decider::COUNT`] deciders at most. The word to the engine
    /// and the fault count of the files `before` stay.
    pub(super) fn of(deciders: usize, before: &Self, leases: impl Fn() -> Leases) -> Self {
        let names = 0..=u8::MAX;
        let files = names
            .take(deciders.max(1))
            .map(|number| File::new(leases(), Some(Decider::new(number))))
            .collect();
        Self {
            files,
            teller: before.teller.clone(),
            faults: AtomicU64::new(before.faults.load(Relaxed)),
        }
    }
}

impl Deref for Files {
    type Target = [File];

    fn deref(&self) -> &[File] {
        &self.files
    }
}

impl MemoryStore {
    /// The frames the ring of a file holds: past them a hand waits until
    /// a holder applied before it writes its own.
    pub const HANDOVER: usize = 1_024;

    /// The file a lease's id names
    /// ([`Blocks::file_of`](super::blocks::Blocks::file_of)): an id of the
    /// leading term by its block's byte, an id of a kept term by that
    /// term's table, any other id by its block modulo the files.
    pub(super) fn file_of(&self, lease: LeaseId) -> usize {
        let files = self.files.len();
        let named = self.blocks.file_of(lease, files);
        if named < files {
            named
        } else {
            0
        }
    }

    /// The file of the decider the calling thread acts as; the first for
    /// a thread that names none and for a name past the files.
    pub(super) fn acting(&self) -> usize {
        let acting = Decider::acting().map_or(0, |decider| usize::from(decider.number()));
        if acting < self.files.len() {
            acting
        } else {
            0
        }
    }

    /// Gives the store the controller's word to the engine: what tells it
    /// of a frame applied to a lease on file. Given once: the first stays.
    #[cfg(any(test, feature = "controller"))]
    pub(crate) fn tell_through(&self, teller: Teller) {
        self.files.teller.get_or_init(|| teller);
    }

    /// A lifecycle call the controller made for a lease panicked and was
    /// held: counted. `true` for the first of a streak, the one its caller
    /// writes a line for.
    #[cfg(any(test, feature = "controller"))]
    pub(crate) fn call_faulted(&self) -> bool {
        let faults = &self.files.faults;
        let first = faults.fetch_or(STREAK, Relaxed) & STREAK == 0;
        faults.fetch_add(1, Relaxed);
        first
    }

    /// A lifecycle call the controller made for a lease returned: a
    /// streak of faults ends. One load where none stands, so a call that
    /// returns writes nothing the deciders share.
    #[cfg(any(test, feature = "controller"))]
    pub(crate) fn call_returned(&self) {
        let faults = &self.files.faults;
        if faults.load(Relaxed) & STREAK != 0 {
            faults.fetch_and(!STREAK, Relaxed);
        }
    }

    /// How many lifecycle calls of a lease panicked and were held: each
    /// cost no request its decision and ended no thread.
    pub fn lifecycle_faults(&self) -> u64 {
        self.files.faults.load(Relaxed) & !STREAK
    }

    /// Tells the engine what `said` did to the lease `record` stood as,
    /// on the file at `at`: under `world` and its decider's name.
    pub(super) fn tell(&self, at: usize, world: &World, record: &LeaseRecord, said: Said) {
        if let Some(tell) = self.files.teller.get() {
            tell(self, world, record, said, self.files[at].name);
        }
    }

    /// The calling thread's seat at its file, at work: a decider's, from
    /// a job at hand to an empty queue. The file is the one of the
    /// decider the thread acts as ([`Decider::acting`]), and one thread
    /// works at a file.
    pub fn at_work(self: &Arc<Self>) -> AtWork {
        // The seat stands before its thread's mark does: a first take
        // that unwinds drops the seat, which clears the mark.
        let mut seat = AtWork {
            store: Arc::clone(self),
            file: self.acting(),
            working: false,
        };
        seat.work();
        seat
    }

    /// The calling thread says it works at the file at `at`, then takes
    /// its turn: the first step of its work. A hand that read nobody at
    /// work and took the turn is through before this thread acts, and
    /// one that takes it behind this step reads the mark under the turn.
    fn begin(&self, at: usize) {
        let was = self.files[at].shown.at_work.swap(me(), SeqCst);
        debug_assert_eq!(was, 0, "one thread works at a file");
        fence(SeqCst);
        drop(self.turn_of(at));
    }

    /// The thread at work at the file at `at` leaves its work. Where it
    /// `looks`, it takes the turn once more: the leaver's half of the
    /// hand-over ([`Self::settle`]).
    fn leave(&self, at: usize, looks: bool) {
        self.files[at].shown.at_work.store(0, SeqCst);
        if looks {
            fence(SeqCst);
            self.settle(at);
        }
    }

    /// The turn of the file at `at`, with everything handed applied where
    /// the caller may act at that file ([`Self::applied`]).
    pub(super) fn turn_of(&self, at: usize) -> Turn<'_> {
        let mut turn = self.files[at].take();
        self.applied(at, &mut turn, &mut Told::default());
        turn
    }

    /// Whether the calling thread may act at the file at `at`: no thread
    /// is at work there, or the caller is that thread. Asked with no turn
    /// held it lets a caller pass a file whose decider works without a
    /// wait for that file's turn; what an apply goes by is read under the
    /// turn ([`Self::applied`]).
    pub(super) fn may_act(&self, at: usize) -> bool {
        let worker = self.files[at].shown.at_work.load(SeqCst);
        worker == 0 || worker == me()
    }

    /// Applies what waits at the file at `at`, under its turn, where the
    /// caller may act as its decider: the thread at work at the file, or
    /// any thread while none is, read under the turn. `false` where
    /// another thread is at work: what waits is that thread's to apply,
    /// so no thread acts as a decider beside it.
    pub(super) fn applied(&self, at: usize, turn: &mut Turn<'_>, told: &mut Told) -> bool {
        if !self.may_act(at) {
            return false;
        }
        let Behind { leases, ring } = &mut *turn.behind;
        while let Ok((gateway, lease, said)) = ring.try_recv() {
            self.apply_frame(at, leases, (gateway, lease), said, told);
        }
        true
    }

    /// Applies what waits at the file at `at` where the caller found no
    /// thread at work, and says whether it did. A hand writes its frame
    /// and then reads who is at work; a thread that leaves work says so
    /// and then comes here. Each puts a fence between its write and its
    /// read, so one of the two sees the other whatever the ring's own
    /// orderings are: the frame is applied by the thread at work or by
    /// its hand. A thread that began work meanwhile is read under the
    /// turn, and what waits is left to it. A thread at work is read
    /// first, with no turn held: its file's turn is not taken from it, and
    /// its next take applies what waits.
    fn settle(&self, at: usize) -> bool {
        if !self.may_act(at) {
            return false;
        }
        let mut turn = self.files[at].take();
        self.applied(at, &mut turn, &mut Told::default())
    }

    /// A stream's end: each file it handed to is looked at once more, so
    /// what it handed is applied before its session leaves, or is left to
    /// the thread at work there. A decider gives the hand of a refused
    /// answer's release back too: it is the thread at work at its file,
    /// so it applies. A hand dropped without this call loses nothing: its
    /// frames are the next take's.
    pub fn hand_back(&self, hand: Hand) {
        for (word, bits) in hand.handed.iter().enumerate() {
            let mut bits = bits.load(Relaxed);
            while bits != 0 {
                let at = word * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if at < self.files.len() {
                    self.settle(at);
                }
            }
        }
    }

    /// Hands the file a lease's id names what a frame said of `lease`: a
    /// session's own hand, and the door of a host that ends its leases
    /// with no stream. The frame is written into the file's ring and
    /// applied by whoever may act at the file next: its decider at work,
    /// or this caller where none is. A full ring stops its caller until a
    /// holder applied, so no frame is dropped: the caller applies where
    /// no thread is at work, and waits for the one that is, which looks
    /// at its ring ahead of each decision. The caller's thread is held
    /// for that wait, one decision of the file's decider at most: it
    /// yields 64 times, then sleeps 50 microseconds between its looks, so
    /// a slow decision costs its waiting hands no core.
    pub fn hand(&self, hand: &Hand, lease: LeaseId, said: Said) {
        let at = self.file_of(lease);
        let file = &self.files[at];
        let mut frame = (hand.gateway, lease, said);
        let mut looks = 0_u32;
        loop {
            match file.shown.hands.try_send(frame) {
                Ok(()) => break,
                Err(TrySendError::Full(back)) => {
                    frame = back;
                    if self.settle(at) {
                        continue;
                    }
                    looks = looks.saturating_add(1);
                    if looks <= FULL_RING_YIELDS {
                        thread::yield_now();
                    } else {
                        thread::sleep(FULL_RING_SLEEP);
                    }
                }
                // The ring's reading end stands behind the file's turn
                // for as long as the store does.
                Err(TrySendError::Disconnected(_)) => return,
            }
        }
        hand.mark(at);
        // The hand's half of the hand-over ([`Self::settle`]).
        fence(SeqCst);
        if file.shown.at_work.load(SeqCst) == 0 {
            self.settle(at);
        }
    }

    /// Applies what one frame said of the lease under `key` on the file
    /// at `at`: the record's read, the door's change and the word to the
    /// engine. A lease not on file is no error and tells nothing: its
    /// dispatch deadline ended it, or its gateway decided that request
    /// itself. Neither does a second dispatch of one lease.
    fn apply_frame(&self, at: usize, leases: &mut Leases, key: Key, said: Said, told: &mut Told) {
        let found = match said {
            Said::Dispatched(dispatch, when) => {
                let found = leases.records.get(&key).map(|held| held.record);
                leases.dispatched(&self.world, &key, dispatch, when);
                found.filter(|record| record.state == LeaseState::Issued)
            }
            Said::Completed(outcome) => {
                let found = leases.records.get(&key).map(|held| held.record);
                leases.completed(&key, &outcome);
                found
            }
            Said::Released(..) => leases.end(&key),
        };
        if let Some(record) = found {
            self.tell(at, told.world(self), &record, said);
        }
    }
}
