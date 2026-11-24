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
//! first step of its work, tries the turn once at each request it takes,
//! whatever its answer, takes it at each issue, takes it behind the
//! give-up of an answer its stream refused, and takes it once more before
//! it waits for its next job. While it is at work no other thread
//! applies at its file: a thread that holds the file's turn and finds its
//! decider at work lets go and applies nothing, and what was handed waits
//! for the decider's next take. So a file's lines stay with the one
//! thread that reads them.
//!
//! A decider that waits for a job keeps its watch while its file holds a
//! lease ([`AtWork::look`]): it looks at the file again within its host's
//! period, and a hand leaves its frame in the ring for that look. The
//! watch starts at a look that finds a lease on file and stands through
//! the jobs behind it. A hand reads it from a word the decider writes
//! only where a watch starts or ends, so a session that hands to a
//! watched file writes the ring and reads nothing a decider writes at a
//! job. The price is a lag on a lull: a frame handed while the file's
//! decider waits is applied one period later at most, and another file's
//! decision reads what its apply writes that much later: its worker's
//! count, high for a lease that ended and low on a target a dispatch moved
//! a lease to, and what a stage keeps of a lease's events. A look says whether
//! it applied a frame ([`LookOutcome`]), and its host sets the time to its next
//! look by that: short while the file's frames flow, longer over a quiet
//! file. A watch ends where a look finds no lease on file: the decider
//! takes its word back and looks once more. A frame that look misses was
//! handed against the old word and names no lease on file: it changes
//! nothing, and it is the next take's. Where another thread lays that
//! lease behind the look, the next take is near: the next hand at the
//! file applies in place and finds the frame first in the ring, and the
//! sweep's next pass applies it otherwise. A thread that rests on its
//! watch ([`AtWork::rest`]), as a decider does where its queue ends,
//! takes the word back over leases on file: a frame a hand left under
//! the old word then names one, and waits for the same takes.
//!
//! A job behind a watch takes the turn first and says it works under it,
//! with the take's apply behind the mark. What the watch kept was handed
//! while nobody worked: a hand at a file with no watch would have
//! applied it in place. So whoever holds the turn and reads the file's
//! thread at work reads it behind that apply. A thread with no watch
//! says it works first and then takes its turn: its hands write a frame
//! and then read who works, and one of the two sees the other.
//!
//! With no watch at a file and no thread at work there, whoever takes
//! the turn applies: the hand itself, as at a file no decider was
//! started for, and the gateways' sweep, which ends the file's leases
//! past their dispatch deadline
//! ([`LeasesStore::expire`](crate::store::LeasesStore::expire)). The
//! sweep applies at a watched file too while its decider waits, and so
//! do a stream's end ([`MemoryStore::hand_back`]) and a hand that finds
//! the ring full. So a frame of a lease on file waits one period at
//! most, but behind a thread that left its work by an unwind or rested
//! on its watch: what it left is the next take's.
//!
//! The hand-over, where a file shows no watch. A frame handed while the
//! file's decider leaves its work waits on no lull: the leaver's last
//! take finds it, or its hand reads the mark the leaver wrote, or a
//! later one. A hand at a watched file reads the watch's word and no
//! mark: its frame waits for the watching thread's next take, and
//! nothing of this argument is asked there. The proof is an argument
//! from the language's memory model. It asks nothing of a processor's
//! own ordering, and no case of the suite tells a fence from none. A
//! hand writes its frame, passes a sequentially consistent fence and
//! reads who is at work ([`MemoryStore::hand`]). A thread that leaves
//! its work writes that nobody is at work, passes such a fence and reads
//! the ring under the turn ([`MemoryStore::settle`]). Two pairs of a
//! write and a read cross. The ring's pair is the queue's tail: a push
//! has moved the tail by an atomic exchange before it returns, and a
//! take reports an empty ring only by a read of the tail. Both are facts
//! of the queue crate's code as read at 0.3.14, not of its written
//! contract: a version that moves reads its push and its pop again. The
//! other pair is the mark of who is at work: the leaver stores it and
//! the hand loads it.
//!
//! Atomics follow the C++20 model, which keeps one total order of every
//! sequentially consistent operation and fence. Where a read takes a
//! value older than a write of the same object, a fence before the read
//! in its thread stands in that order before a fence behind the write in
//! its thread. Suppose both reads missed. The take read the tail as it
//! stood before the hand's push, so the leaver's fence stands before the
//! hand's. The hand read the leaver still at work, a value older than
//! the leaver's store, so the hand's fence stands before the leaver's.
//! One order holds no such two fences, so one read sees the other's
//! write. A take that reads a moved tail waits for the slot's stamp and
//! takes the frame. A hand that reads nobody at work takes the turn and
//! applies what waits; where a thread began work since, the frame is
//! left to that thread's takes. That thread wrote its mark, passed a
//! fence of its own and took the turn ([`MemoryStore::begin`]): its
//! first take finds the frame of a hand that missed the mark, and a
//! later one the frame of a hand that read it. A thread that begins
//! behind its watch holds the turn already as it writes the mark and
//! passes the fence, and applies under it
//! ([`MemoryStore::begin_watched`]): that apply stands where the other's
//! first take does, for a hand that read no word yet. None of it leans
//! on the orderings the queue gives its own reads and writes.
//!
//! A decision reads its own file behind everything handed to it. A row
//! that another file's leases book lags by that file's next take: a
//! worker's count is one atomic a row, stepped under the turn of the file
//! that books it, and a stage's own book of a lease's events is written
//! where the lease's frame is applied. A stage that places by either
//! places elsewhere. A stage that may refuse on a count it is shown, or
//! on what the leases' events told it, has its decision first apply what
//! waits ([`AtWork::apply_waiting`]): at its own file by a take that
//! waits, and at each other file that shows a watch. A stage that says
//! nothing is taken as one: a plugin's, whose rule is its own, and a
//! host's own stage. Each stage of this crate says it is none. The day
//! the reservations' `try_reserve` gets a caller, or a stage of this
//! crate sheds on a count or on its book, it takes the same entry.
//!
//! The lock order, written here once: a file's turn is taken first or not
//! at all, and one turn at a time. The term's rise holds every turn, in
//! the files' order (`set_term`, `follow_term`), and so does a standby's
//! restore, from its start until it has read the ids on file
//! ([`MemoryStore::restore_from`]). Under a turn come the world's load
//! (no lock), a stage's own lock, the engine's sessions table, the
//! journal's word of a term, the table of the members no world lists yet,
//! a leaf lock, and the entry of the worker objects a host's word to the
//! engine asks, a leaf lock too. A standby's restore takes the
//! store's tables under its turns too: the workers', the health's and the
//! loads', in that order. No path takes one of those and then a turn. A
//! lifecycle call of a served controller therefore runs under a turn, or
//! after it where a decision gives its own lease up: it is bounded and
//! calls no door of the leases. A registration that takes its key's pins
//! along holds the workers lock over the engine's sessions table, each
//! shard a leaf under it. A debug build marks the thread that holds a turn
//! and refuses a second take of it by that thread.

use std::{
    fmt,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{
            fence, AtomicBool, AtomicU64, AtomicUsize,
            Ordering::{Relaxed, SeqCst},
        },
        Arc, OnceLock,
    },
    thread,
    time::Duration,
};

use arc_swap::Guard;
use parking_lot::{Mutex, MutexGuard};

use super::{takes_dispatch, Key, Leases, MemoryStore};
#[cfg(feature = "controller")]
use crate::tracking::lease;
use crate::{
    engine::decision_slot::DecisionSlot,
    store::{
        leases::{LeaseState, Sides},
        snapshot::StoreSnapshot,
        GatewayId,
    },
    tracking::lease::{LeaseId, Said},
};

/// What tells the engine of a frame applied to a lease on file: the world
/// the frame is told under, the lease as it stood before the frame, both
/// its parts, what the frame said, and the decider whose file the lease is
/// on, where the store's host named its deciders. It holds what its host
/// gave it: a served controller's holds the replica's worker objects.
type Teller =
    Arc<dyn Fn(&MemoryStore, &StoreSnapshot, &Sides, Said, Option<DecisionSlot>) + Send + Sync>;

/// One load of the world for a run of words to the engine: the store's
/// world, loaded at the run's first word and held to its last. A run is
/// one loop's words: the frames an apply finds in the ring, the leases an
/// issue ends, the leases a sweep ends in a file. So a run that tells
/// nothing loads no world, and one that tells many loads one: each word
/// of a run is told under the generation that stood at the run's first.
/// Each run builds its own where it tells, so one hold of a turn may load
/// twice: the leases an issue ends may be told under a later world than
/// the frames its apply found. Where a generation starts between the two,
/// the engine that issued those leases does not hear their ends. That is
/// a limit of this form, not its aim: one load for both runs undoes it.
/// None is passed in by a caller: that cost every take a holder's
/// building and its drop, whatever the take found.
#[derive(Default)]
pub(super) struct OneLoad(Option<Guard<Arc<StoreSnapshot>>>);

impl OneLoad {
    /// The world this run tells under.
    pub(super) fn world(&mut self, store: &MemoryStore) -> &StoreSnapshot {
        self.0.get_or_insert_with(|| store.world.load())
    }
}

/// What waits in a file's ring of a lease: what a frame said of it, or the
/// served door's own word of a decode call's issue. No frame carries the
/// second and no host says it: one function of the crate writes it
/// ([`MemoryStore::hand_taken_back`]).
#[derive(Clone, Copy)]
enum Word {
    /// What a frame said, or a stream's release.
    Said(Said),
    /// One issue of a decode call taken back, and the reason.
    #[cfg(feature = "controller")]
    TakenBack(super::DecodeIssue, lease::ReleaseReason),
}

// The door's own word rides the room a frame's word leaves: a slot of the
// ring is no wider for it.
const _: () = assert!(size_of::<Word>() == size_of::<Said>());

/// A word of a gateway's lease, as it waits in the ring.
type Handed = (GatewayId, LeaseId, Word);

/// A file's ring by its two ends. They stand in a module of their own
/// and the queue is private to it: the turn module writes a frame through
/// the writing end, takes one through the reading end it holds
/// exclusively, and reaches the queue no other way.
mod ring {
    use std::sync::Arc;

    use crossbeam_queue::ArrayQueue;

    use super::{Handed, MemoryStore};

    /// A file's ring by its two ends: [`MemoryStore::HANDOVER`] frames
    /// written in place and taken in the order they were handed. The
    /// ends are two types: a hand writes a frame and can take none.
    pub(super) fn ends() -> (Hands, Taker) {
        let frames = Arc::new(ArrayQueue::new(MemoryStore::HANDOVER));
        (Hands(Arc::clone(&frames)), Taker(frames))
    }

    /// The ring's writing end, every hand's.
    pub(super) struct Hands(Arc<ArrayQueue<Handed>>);

    impl Hands {
        /// Writes `frame` behind what waits, or gives it back where the
        /// ring is full. It wakes no thread: none sleeps at a ring, and
        /// a frame is applied by a thread that looks
        /// ([`MemoryStore::hand`]).
        pub(super) fn write(&self, frame: Handed) -> Result<(), Handed> {
            self.0.push(frame)
        }

        /// Whether a frame waits for the next take.
        pub(super) fn waits(&self) -> bool {
            !self.0.is_empty()
        }
    }

    /// The ring's reading end. It stands behind the file's turn, so the
    /// turn's holder alone takes a frame.
    pub(super) struct Taker(Arc<ArrayQueue<Handed>>);

    impl Taker {
        /// The frame handed first of those that wait.
        pub(super) fn take(&mut self) -> Option<Handed> {
            self.0.pop()
        }

        /// Whether a frame waits: a look's read, ahead of its apply
        /// ([`MemoryStore::looked`]).
        pub(super) fn waits(&self) -> bool {
            !self.0.is_empty()
        }
    }
}

/// What a hand reads of a file at each frame, on lines of its own: no
/// thread writes them while the file's decider keeps its watch.
#[repr(align(128))]
struct Shown {
    /// The ring's writing end, every hand's.
    hands: ring::Hands,
    /// Whether the file's decider keeps its watch over the ring: with a
    /// job at hand or none, it takes what the ring holds within its
    /// host's period ([`AtWork::look`]). A hand that reads `true` leaves
    /// its frame in the ring. Written where a watch starts and where it
    /// ends.
    watched: AtomicBool,
}

/// Who is at work at a file, in a block of its own. A hand at a file with
/// no watch reads it at each frame, and the file's decider writes it at
/// each job that ends a wait. So it stands off the lines every hand reads
/// ([`Shown`]), which no decider writes at a job. It stands off the lines
/// behind the turn too, which the turn's holder writes at each issue and
/// each apply: on one line with the last of them, a hand's read met a
/// write of each decision, and a controller's only decider served 6 to 8
/// per cent fewer by a timed pair.
#[repr(align(128))]
struct Working {
    /// The mark of the thread at work at the file ([`me`]); 0 for none.
    /// One thread at most works at a file: its decider, with a job at
    /// hand. It applies what is handed while it works, and once more as
    /// it leaves.
    at_work: AtomicUsize,
}

/// What stands behind the turn: the file, and the reading end of its
/// ring.
pub(super) struct Behind {
    leases: Leases,
    ring: ring::Taker,
}

/// One decider's file of leases behind its turn, with the ring of what is
/// handed to it.
pub(super) struct File {
    /// The turn: its holder writes the file.
    turn: Mutex<Behind>,
    /// When a sweep next asks the file, written under `turn`
    /// ([`super::NextDue`]).
    next_due: Arc<super::NextDue>,
    /// What a hand reads at each frame.
    shown: Shown,
    /// Who is at work at the file: what a hand reads where no watch
    /// stands.
    working: Working,
    /// The decider whose file this is, where the store's host named its
    /// deciders: a lease's events are told under it.
    name: Option<DecisionSlot>,
    /// The thread that holds the turn; 0 for none.
    #[cfg(debug_assertions)]
    holder: AtomicUsize,
    /// Takes of this file's turn, for cases that assert a path avoided it.
    #[cfg(test)]
    takes: AtomicUsize,
}

impl File {
    fn new(leases: Leases, name: Option<DecisionSlot>) -> Self {
        let (hands, ring) = ring::ends();
        let next_due = Arc::clone(&leases.next_due);
        Self {
            turn: Mutex::new(Behind { leases, ring }),
            next_due,
            shown: Shown {
                hands,
                watched: AtomicBool::new(false),
            },
            working: Working {
                at_work: AtomicUsize::new(0),
            },
            name,
            #[cfg(debug_assertions)]
            holder: AtomicUsize::new(0),
            #[cfg(test)]
            takes: AtomicUsize::new(0),
        }
    }

    /// Whether this file may hold work for a sweep.
    pub(super) fn sweep_needed(&self, now: std::time::Instant) -> bool {
        self.next_due.reached(now) || self.shown.hands.waits()
    }

    /// Takes the turn and applies nothing: for the term's read, which no
    /// handed frame changes.
    pub(super) fn take(&self) -> Turn<'_> {
        #[cfg(test)]
        self.takes.fetch_add(1, SeqCst);
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

    /// Takes of this file's turn since the test last cleared them.
    #[cfg(test)]
    pub(super) fn takes(&self) -> usize {
        self.takes.load(SeqCst)
    }

    /// Starts this test's count of turn takes again.
    #[cfg(test)]
    pub(super) fn clear_takes(&self) {
        self.takes.store(0, SeqCst);
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

/// What a look says of its thread's watch ([`AtWork::look`]): whether it
/// stands, and whether the look applied a frame. Its host sets the time
/// to its next look by it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LookOutcome {
    /// The thread keeps no watch: the file holds no lease, and a watch
    /// it kept has ended, by its word first and one more look behind it.
    /// The hands apply in place, and the thread may wait with no bound.
    /// Said too where another thread is at work at the file: what waits
    /// there is that thread's.
    Ended,
    /// A lease stands on the file and the look applied no frame: the
    /// watch stands over a quiet file. The hands leave their frames in
    /// the ring, and the thread owes the file its next look, by
    /// [`AtWork::look`] or by its next job.
    Quiet,
    /// A lease stands on the file and the look applied a frame: the
    /// watch stands as under [`LookOutcome::Quiet`], over a file whose frames
    /// flow.
    Fed,
}

/// A decider's place at its file ([`MemoryStore::at_work`]). While its
/// thread works the hands leave what they hand to that file to it: it
/// applies where it finds its turn free at each request it takes
/// ([`AtWork::apply`]), at each issue, and behind the give-up of an
/// answer its stream refused. A thread with no job at hand looks at its
/// file ([`AtWork::look`]) and keeps its watch while the file holds a
/// lease: the hands still leave their frames to it, and it owes the file
/// its next look within its host's period. A thread that rests
/// ([`AtWork::rest`]) or drops its place leaves its work, ends a watch it
/// keeps and takes the turn once more: what was handed behind its last
/// take is applied before the thread waits, and the hands apply in place
/// from there. A thread that rests behind each job never looks, and
/// keeps no watch.
pub struct AtWork {
    store: Arc<MemoryStore>,
    file: usize,
    working: bool,
    /// Whether the file shows this thread's watch ([`Shown::watched`]):
    /// from a look that found a lease on file to the watch's end.
    watching: bool,
}

impl AtWork {
    /// Applies what is handed where the turn is free: a decider's look at
    /// each request it takes, ahead of the rows' counts a decision reads.
    /// Where the turn is held the request goes on and nothing is applied:
    /// what is handed waits for the next take that applies. A decision's
    /// issue is one, and a refused request has none. Another file's
    /// decision under a stage that may refuse by a report of leases
    /// holds this turn for one read ([`Self::apply_waiting`]), as the
    /// sweep holds it for a pass: a look that meets either misses, and
    /// this thread's own decision under such a stage takes the turn
    /// again by a wait.
    pub fn apply(&self) {
        if let Some(mut turn) = self.store.files[self.file].try_take() {
            self.store.applied(self.file, &mut turn);
        }
    }

    /// Applies what waits at this thread's own file and at each other
    /// file that shows a watch, for a decision under a stage that may
    /// refuse on a count it is shown or on its book of the leases'
    /// events: the stage then reads what a hand would have applied in
    /// place. Its own file comes first, by a take that waits: the
    /// thread's look at this request tried the turn once
    /// ([`Self::apply`]), and another file's decision under such a stage
    /// may have held it then. It then takes each watched file's turn,
    /// whatever the file's thread does. Where no thread works there this
    /// caller applies, as any thread may. A thread that took up a job
    /// behind its watch said so under the turn, with its apply behind
    /// the mark ([`AtWork::work`]): read at work under the turn, it has
    /// applied what its watch kept, and what is handed from there is its
    /// own, as a hand leaves it. Each wait is one take of a file, an
    /// apply or an issue, never a job.
    pub fn apply_waiting(&self) {
        drop(self.store.turn_of(self.file));
        let files = 0..self.store.files.len();
        for at in files.filter(|at| *at != self.file) {
            // A decision's read, taken in the one order of the watch's
            // writes. A file that shows no watch holds nothing for it:
            // its hands applied in place, or left their frames to a
            // thread at work there, and a watch that ended found no
            // lease on file, so a frame its last look missed moves no
            // count. Its turn is not taken.
            if self.store.files[at].shown.watched.load(SeqCst) {
                drop(self.store.turn_of(at));
            }
        }
    }

    /// The thread has a job at hand again: it says so and takes its
    /// turn, the first step of its work. A watch it keeps stands through
    /// the job, and the job starts none: the hands read a thread with no
    /// watch as the thread at work. With no watch the thread says it
    /// works and then takes the turn. Behind a watch the turn comes
    /// first, and the thread says it works under it with the take's
    /// apply behind the mark: a decision at another file that reads this
    /// thread at work under the turn reads behind what its watch kept
    /// ([`AtWork::apply_waiting`]).
    pub fn work(&mut self) {
        if !self.working {
            self.working = true;
            if self.watching {
                self.store.begin_watched(self.file);
            } else {
                self.store.begin(self.file);
            }
        }
    }

    /// The thread has no job at hand: it leaves its work, applies what
    /// was handed behind its last take and says what stands of its watch
    /// ([`LookOutcome`]). While its file holds a lease the watch stands, and
    /// starts here where the thread kept none: the hands leave their
    /// frames in the ring, and the thread owes the file its next look
    /// within its host's period, by this call or by its next job. Where
    /// the file holds none the watch has ended, by its word first and one
    /// more look behind it: the hands apply in place, and the thread may
    /// wait with no bound.
    #[must_use = "a thread that keeps its watch owes its file the next look"]
    pub fn look(&mut self) -> LookOutcome {
        if self.working {
            self.working = false;
            self.store.leave(self.file);
        }
        match self.store.looked(self.file) {
            // A watch ends by its word first and one more look. A thread
            // that kept none has no word to take back.
            LookOutcome::Ended if self.watching => self.end_watch(),
            LookOutcome::Ended => LookOutcome::Ended,
            stands => {
                if !self.watching {
                    // A hand that read no word read who is at work behind
                    // its fence: it applied in place, or the look above
                    // applied its frame. The watch starts with nothing
                    // left behind it.
                    self.watching = true;
                    self.word().store(true, SeqCst);
                }
                stands
            }
        }
    }

    /// Ends the watch of a thread whose look found no lease on file, and
    /// says what stands of it after all. The word is taken back first and
    /// the file looked at once more behind it. A hand that read the old
    /// word left its frame in the ring: this look applies it, or it finds
    /// no lease on file and the frame names none. A lease it finds, laid
    /// by another thread since the first look, keeps the watch: its
    /// frames may wait in the ring.
    pub(crate) fn end_watch(&mut self) -> LookOutcome {
        self.word().store(false, SeqCst);
        fence(SeqCst);
        let found = self.store.looked(self.file);
        if found == LookOutcome::Ended {
            self.watching = false;
        } else {
            self.word().store(true, SeqCst);
        }
        found
    }

    /// The thread waits for its next job and owes its file no look: it
    /// ends a watch it keeps and says it left its work, and what was
    /// handed behind its last take is applied before it waits. The hands
    /// apply in place from there. A thread that kept no watch leaves no
    /// frame behind: its hands read who is at work, with a fence on both
    /// sides. One that ends a watch here takes the word back over leases
    /// on file: a hand that read the old word may leave a frame of one
    /// behind this take, for the file's next: the next hand's, which
    /// applies in place, or the sweep's pass.
    pub fn rest(&mut self) {
        if self.working || self.watching {
            self.working = false;
            if self.watching {
                self.watching = false;
                self.word().store(false, SeqCst);
            }
            self.store.leave(self.file);
            self.store.settle(self.file);
        }
    }

    /// The word of this place's watch, as its file shows it.
    fn word(&self) -> &AtomicBool {
        &self.store.files[self.file].shown.watched
    }
}

impl Drop for AtWork {
    /// A thread that leaves by an unwind looks at nothing: what it left
    /// is the next take's, and a tell that unwound is not asked again
    /// under it. A watch it keeps ends either way, as at a rest
    /// ([`AtWork::rest`]): the hands apply in place.
    fn drop(&mut self) {
        if self.working || self.watching {
            if self.watching {
                self.word().store(false, SeqCst);
            }
            self.store.leave(self.file);
            if !thread::panicking() {
                self.store.settle(self.file);
            }
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
    handed: [AtomicU64; DecisionSlot::COUNT / 64],
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
    /// names [`DecisionSlot::COUNT`] deciders at most. The word to the engine
    /// and the fault count of the files `before` stay.
    pub(super) fn of(deciders: usize, before: &Self, leases: impl Fn() -> Leases) -> Self {
        let names = 0..=u8::MAX;
        let files = names
            .take(deciders.max(1))
            .map(|number| File::new(leases(), Some(DecisionSlot::new(number))))
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
    pub(crate) fn file_of(&self, lease: LeaseId) -> usize {
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
        let acting = DecisionSlot::acting().map_or(0, |decider| usize::from(decider.number()));
        if acting < self.files.len() {
            acting
        } else {
            0
        }
    }

    /// Gives the store the controller's word to the engine: what tells it
    /// of a frame applied to a lease on file. Given once: the first stays.
    #[cfg_attr(
        not(any(test, feature = "controller")),
        expect(dead_code, reason = "called by the served controller alone")
    )]
    pub(crate) fn tell_through(
        &self,
        teller: impl Fn(&MemoryStore, &StoreSnapshot, &Sides, Said, Option<DecisionSlot>)
            + Send
            + Sync
            + 'static,
    ) {
        self.files.teller.get_or_init(|| Arc::new(teller));
    }

    /// A lifecycle call the controller made for a lease panicked and was
    /// held: counted. `true` for the first of a streak, the one its caller
    /// writes a line for.
    #[cfg_attr(
        not(any(test, feature = "controller")),
        expect(dead_code, reason = "called by the served controller alone")
    )]
    pub(crate) fn call_faulted(&self) -> bool {
        let faults = &self.files.faults;
        let first = faults.fetch_or(STREAK, Relaxed) & STREAK == 0;
        faults.fetch_add(1, Relaxed);
        first
    }

    /// A lifecycle call the controller made for a lease returned: a
    /// streak of faults ends. One load where none stands, so a call that
    /// returns writes nothing the deciders share.
    #[cfg_attr(
        not(any(test, feature = "controller")),
        expect(dead_code, reason = "called by the served controller alone")
    )]
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

    /// Tells the engine what `said` did to the lease `found` stood as, on
    /// the file at `at`: under `world` and its decider's name.
    pub(super) fn tell(&self, at: usize, world: &StoreSnapshot, found: &Sides, said: Said) {
        if let Some(tell) = self.files.teller.get() {
            tell(self, world, found, said, self.files[at].name);
        }
    }

    /// How many files show a watch, for the cases: no door of the store
    /// tells it.
    #[cfg(test)]
    pub(crate) fn watches(&self) -> usize {
        let files = self.files.iter();
        files.filter(|file| file.shown.watched.load(SeqCst)).count()
    }

    /// How many files show a thread at work, for the deciders' cases: no
    /// door of the store tells it.
    #[cfg(all(test, feature = "controller"))]
    pub(crate) fn working(&self) -> usize {
        let files = self.files.iter();
        files
            .filter(|file| file.working.at_work.load(SeqCst) != 0)
            .count()
    }

    /// The calling thread's place at its file, at work: a decider's, from
    /// a job at hand to an empty queue. The file is the one of the
    /// decider the thread acts as ([`DecisionSlot::acting`]), and one thread
    /// at a time works at a file.
    pub fn at_work(self: &Arc<Self>) -> AtWork {
        // The place stands before its thread's mark does: a first take
        // that unwinds drops the place, which clears the mark.
        let mut place = AtWork {
            store: Arc::clone(self),
            file: self.acting(),
            working: false,
            watching: false,
        };
        place.work();
        place
    }

    /// The calling thread says it works at the file at `at`, then takes
    /// its turn: the first step of its work. A hand that read nobody at
    /// work and took the turn is through before this thread acts, and
    /// one that takes it behind this step reads the mark under the turn.
    /// A thread that keeps its watch over the file begins by
    /// [`Self::begin_watched`].
    fn begin(&self, at: usize) {
        let was = self.files[at].working.at_work.swap(me(), SeqCst);
        debug_assert_eq!(was, 0, "one thread works at a file");
        fence(SeqCst);
        drop(self.turn_of(at));
    }

    /// [`Self::begin`] for a thread that keeps its watch over the file
    /// at `at`. Its hands read the word and not who works, so what the
    /// watch kept was handed while nobody worked: a hand at a file with
    /// no watch would have applied it in place, and a decision that may
    /// refuse by a report of leases reads behind it
    /// ([`AtWork::apply_waiting`]). So the turn comes first, and the
    /// mark is written under it with the take's apply behind its fence:
    /// whoever holds the turn and reads this thread at work reads it
    /// behind that apply. The fence is the worker's half of the
    /// hand-over all the same ([`Self::settle`]): a hand that read no
    /// word yet, and nobody at work, wrote its frame ahead of the mark.
    fn begin_watched(&self, at: usize) {
        let file = &self.files[at];
        let mut turn = file.take();
        let was = file.working.at_work.swap(me(), SeqCst);
        debug_assert_eq!(was, 0, "one thread works at a file");
        fence(SeqCst);
        self.applied(at, &mut turn);
    }

    /// The thread at work at the file at `at` leaves its work: it takes
    /// its mark back, with a fence behind the write. The take that
    /// follows, its look ([`Self::looked`]) or its last apply, is the
    /// leaver's half of the hand-over ([`Self::settle`]).
    fn leave(&self, at: usize) {
        self.files[at].working.at_work.store(0, SeqCst);
        fence(SeqCst);
    }

    /// A look at the file at `at` by its own decider with no job at hand:
    /// what waits is applied under the turn, and the answer is what
    /// stands of the watch behind it: whether the file holds a lease, and
    /// whether this look applied a frame. The look reads that from the
    /// ring under its turn, ahead of the apply: the apply is every
    /// take's, and it counts no frame. A frame handed while the look
    /// applies is applied all the same and not said: the time to the
    /// next look then grows once where it would have fallen.
    /// [`LookOutcome::Ended`] too where another thread is at work there: what
    /// waits is that thread's.
    fn looked(&self, at: usize) -> LookOutcome {
        if !self.may_act(at) {
            return LookOutcome::Ended;
        }
        let mut turn = self.files[at].take();
        let fed = turn.behind.ring.waits();
        if !self.applied(at, &mut turn) {
            return LookOutcome::Ended;
        }
        if turn.records.is_empty() {
            LookOutcome::Ended
        } else if fed {
            LookOutcome::Fed
        } else {
            LookOutcome::Quiet
        }
    }

    /// The turn of the file at `at`, with everything handed applied where
    /// the caller may act at that file ([`Self::applied`]).
    pub(super) fn turn_of(&self, at: usize) -> Turn<'_> {
        let mut turn = self.files[at].take();
        self.applied(at, &mut turn);
        turn
    }

    /// Whether the calling thread may act at the file at `at`: no thread
    /// is at work there, or the caller is that thread. Asked with no turn
    /// held it lets a caller pass a file whose decider works without a
    /// wait for that file's turn; what an apply goes by is read under the
    /// turn ([`Self::applied`]).
    pub(super) fn may_act(&self, at: usize) -> bool {
        let worker = self.files[at].working.at_work.load(SeqCst);
        worker == 0 || worker == me()
    }

    /// Applies what waits at the file at `at`, under its turn, where the
    /// caller may act as its decider: the thread at work at the file, or
    /// any thread while none is, read under the turn. `false` where
    /// another thread is at work: what waits is that thread's to apply,
    /// so no thread acts as a decider beside it. The frames it finds are
    /// told under one load of the world, the apply's own ([`OneLoad`]).
    pub(super) fn applied(&self, at: usize, turn: &mut Turn<'_>) -> bool {
        if !self.may_act(at) {
            return false;
        }
        let Behind { leases, ring } = &mut *turn.behind;
        let mut one = OneLoad::default();
        while let Some((gateway, lease, word)) = ring.take() {
            let key = (gateway, lease);
            match word {
                Word::Said(said) => self.apply_frame(at, leases, key, said, &mut one),
                #[cfg(feature = "controller")]
                Word::TakenBack(issue, reason) => {
                    self.apply_taken_back(at, leases, key, issue, reason, &mut one);
                }
            }
        }
        true
    }

    /// Applies what waits at the file at `at` where the caller found no
    /// thread at work, and says whether it did. A hand at a file with no
    /// watch writes its frame and then reads who is at work; a thread
    /// that leaves work says so and then comes here. Each puts a fence
    /// between its write and its read, so one of the two sees the other
    /// whatever the ring's own orderings are: the frame is applied by the
    /// thread at work or by its hand. A thread that began work meanwhile
    /// is read under the turn, and what waits is left to it. A thread at
    /// work is read first, with no turn held: its file's turn is not
    /// taken from it, and its next take applies what waits. A thread that
    /// begins behind its watch writes its mark under the turn, with the
    /// same fence ahead of its apply ([`Self::begin_watched`]): a hand
    /// that read no word yet is one of the two there as well.
    ///
    /// Kept out of line: laid into a hand it costs each hand three
    /// instructions, by count.
    #[inline(never)]
    fn settle(&self, at: usize) -> bool {
        if !self.may_act(at) {
            return false;
        }
        let mut turn = self.files[at].take();
        self.applied(at, &mut turn)
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
    /// session's own hand. The frame is written into the file's
    /// ring. Where the file's decider keeps its watch the hand leaves it there:
    /// the decider applies it at its next take, one period of its host away at
    /// most. The word is read with no fence behind the ring's write, so a hand
    /// can read the old word behind a watch's last look: its frame then names
    /// no lease on file, and waits for the file's next take. Where another
    /// thread lays that lease behind the look, that take is the next hand's,
    /// which applies in place, or the sweep's next pass. Behind a thread that
    /// rested on its watch ([`AtWork::rest`]) such a frame may name a lease on
    /// file, and waits for the same takes. With no watch a frame is applied by
    /// whoever may act at the file next: a thread at work there, or this caller
    /// where none is.
    ///
    /// A full ring stops its caller until a holder applied, so no frame
    /// is dropped: the caller applies where no thread is at work, and
    /// waits for the one that is, which looks at its ring at each request
    /// it takes, whatever its answer. The caller's thread is held for
    /// that wait, to the first such look that finds the file's turn free:
    /// it yields `FULL_RING_YIELDS` times, then sleeps `FULL_RING_SLEEP`
    /// between its looks, so a slow decision costs its waiting hands no
    /// core.
    pub fn hand(&self, hand: &Hand, lease: LeaseId, said: Said) {
        self.hand_word(hand, lease, Word::Said(said));
    }

    /// Hands the file `lease`'s id names the take-back of `issue`, one
    /// issue of a decode call, under `reason`: the word of a `Decide`
    /// stream for an answer of that call that does not leave. It goes the
    /// way of [`Self::hand`], so whoever may act at the file applies it,
    /// and no stage hears it beside its decider at work. Where that issue
    /// still stands the side returns to chosen and its engine hears the
    /// decode side's release. Where the side moved since, by a dispatch, a
    /// repeat's take-back or a later issue, nothing moves and nothing is
    /// told.
    #[cfg(feature = "controller")]
    #[cold]
    pub(crate) fn hand_taken_back(
        &self,
        hand: &Hand,
        lease: LeaseId,
        issue: super::DecodeIssue,
        reason: lease::ReleaseReason,
    ) {
        self.hand_word(hand, lease, Word::TakenBack(issue, reason));
    }

    /// [`Self::hand`] for either word of the ring, laid into it: a frame's
    /// word keeps the loop the served hand was, and the door's own word
    /// pays its copy on a path few answers take.
    #[inline(always)]
    fn hand_word(&self, hand: &Hand, lease: LeaseId, word: Word) {
        let at = self.file_of(lease);
        let mut looks = 0_u32;
        while !self.try_hand_word(at, hand, lease, word) {
            looks = looks.saturating_add(1);
            if looks <= FULL_RING_YIELDS {
                thread::yield_now();
            } else {
                thread::sleep(FULL_RING_SLEEP);
            }
        }
    }

    /// Writes a word of `lease` into the file's ring and marks the hand.
    /// `false` where the ring is full and stays so: a thread at work at the
    /// file keeps what its ring holds.
    #[inline(always)]
    fn try_hand_word(&self, at: usize, hand: &Hand, lease: LeaseId, word: Word) -> bool {
        let file = &self.files[at];
        let frame = (hand.gateway, lease, word);
        loop {
            match file.shown.hands.write(frame) {
                Ok(()) => break,
                // The caller applies where no thread is at work, and
                // looks again: a thread at work keeps what its ring holds.
                Err(_) if self.settle(at) => {}
                Err(_) => return false,
            }
        }
        hand.mark(at);
        // A watched file: the frame is its decider's. The word shares its
        // line with the ring's end alone, and no decider writes it at a
        // job.
        if file.shown.watched.load(Relaxed) {
            return true;
        }
        // The hand's half of the hand-over ([`Self::settle`]).
        fence(SeqCst);
        if file.working.at_work.load(SeqCst) == 0 {
            self.settle(at);
        }
        true
    }

    /// Applies what one frame said of the lease under `key` on the file
    /// at `at`: the record's read, the door's change and the word to the
    /// engine. A lease not on file is no error and tells nothing: its
    /// dispatch deadline ended it, or its gateway decided that request
    /// itself. Neither does a dispatch on a side an earlier dispatch
    /// moved, nor a frame on a side of a pair's lease that ended: the word
    /// is of the sides the record held as the frame found it. A dispatch
    /// at an index a chosen side's pick holds no target at moves nothing
    /// and is handed on all the same.
    fn apply_frame(&self, at: usize, leases: &mut Leases, key: Key, said: Said, one: &mut OneLoad) {
        let found = match said {
            Said::Dispatched(dispatch, when) => leases
                .dispatched(self.slots(), &key, dispatch, when)
                .filter(|found| takes_dispatch(found, dispatch.side)),
            Said::Completed(outcome) => leases.end_side(&key, outcome.side, LeaseState::Completed),
            Said::Released(side, reason) => leases.released(&key, side, reason),
        };
        if let Some(found) = found {
            self.tell(at, one.world(self), &found, said);
        }
    }
}
