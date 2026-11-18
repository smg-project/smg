//! The turn of the leases' file: who holds it writes the file, and what a
//! session's frame says of a lease is handed to the file, never written by
//! the session.
//!
//! A session reads its frame whole and hands what it says to the file's
//! ring ([`MemoryStore::hand`]). Whoever takes the turn applies everything
//! handed before it reads a record, ends a dispatch deadline or issues:
//! the record's read, the door's change and the word to the engine, in the
//! order the frames were handed, so one lease's events are told in their
//! order whoever applies them. The ring is written in place and holds
//! [`MemoryStore::HANDOVER`] frames: a session that finds it full takes
//! the turn and applies, so nothing handed is dropped and the ring never
//! grows.
//!
//! Who applies. A decider with a job at hand is at work at the file
//! ([`MemoryStore::at_work`]): it tries the turn once before each
//! decision reads its rows' counts, takes it at each issue, and takes it
//! once more before it waits for its next job. A turn it finds held is
//! its holder's to apply under: a decider never waits for the turn ahead
//! of a decision. While any is at work a session leaves what it handed
//! to them, so the file's lines stay with the threads that read them: a
//! handed frame waits for the next take of a thread at work, a
//! decision's try or its issue. With none at work the session takes the
//! turn and applies. So no handed frame waits on a lull.
//!
//! The lock order, written here once: the turn is taken first or not at
//! all. Under it come the world's load (no lock), a stage's own lock, the
//! engine's sessions table, the store's pins and the journal's word of a
//! term. No path takes one of those and then the turn. A lifecycle call
//! of a served controller therefore runs under the turn: it is bounded
//! and calls no door of the leases. A debug build marks the thread that
//! holds the turn and refuses a second take by it.
//!
//! The file is one today, shared by the deciders: the turn is held for an
//! apply, for an issue and for a lease's giving up, never across a
//! decision. A lease's id names its file through one function
//! ([`MemoryStore::file_of`]).

use std::{
    fmt,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{fence, AtomicUsize, Ordering::SeqCst},
        mpsc::{sync_channel, Receiver, SyncSender, TrySendError},
        Arc, OnceLock,
    },
    thread,
};

use parking_lot::{Mutex, MutexGuard};

use super::{Held, Key, Leases, MemoryStore};
use crate::{
    decider::Decider,
    lease::LeaseId,
    store::{
        leases::{LeaseRecord, LeaseState, Said},
        GatewayId,
    },
};

/// What tells the engine of a frame applied to a lease on file: the record
/// as it stood before the frame, what the frame said, and the decider that
/// issued the lease, where its thread named itself.
type Teller = fn(&MemoryStore, &LeaseRecord, Said, Option<Decider>);

/// What a frame said of a gateway's lease, as it waits in the ring.
type Handed = (GatewayId, LeaseId, Said);

/// A count on lines of its own: the threads that write it share no line
/// with the threads that only read its neighbours.
#[derive(Debug, Default)]
#[repr(align(128))]
struct Apart(AtomicUsize);

/// What stands behind the turn: the file, and the reading end of its
/// ring.
pub(super) struct Behind {
    leases: Leases,
    ring: Receiver<Handed>,
}

/// The leases' file behind its turn, with the ring of what is handed to
/// it.
pub(super) struct File {
    /// The turn: its holder writes the file.
    turn: Mutex<Behind>,
    /// The ring's writing end: each hand holds one of its own.
    hands: SyncSender<Handed>,
    /// The threads at work at the file: the deciders with a job at hand.
    /// Each applies what is handed while it works, and once more as it
    /// leaves.
    at_work: Apart,
    /// The controller's word to the engine, given once.
    teller: OnceLock<Teller>,
    /// The thread that holds the turn; 0 for none.
    #[cfg(debug_assertions)]
    holder: AtomicUsize,
}

impl File {
    pub(super) fn new(leases: Leases) -> Self {
        let (hands, ring) = sync_channel(MemoryStore::HANDOVER);
        Self {
            turn: Mutex::new(Behind { leases, ring }),
            hands,
            at_work: Apart::default(),
            teller: OnceLock::new(),
            #[cfg(debug_assertions)]
            holder: AtomicUsize::new(0),
        }
    }

    /// Takes the turn and applies nothing: for the term's read and a
    /// count's, which no handed frame changes.
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

    /// Whether a frame waits in the ring or is about to be written.
    /// The turn where it is free. `None` where a thread holds it: that
    /// thread's take applied what was handed before it.
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
#[cfg(debug_assertions)]
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

/// A thread at work at the leases' file ([`MemoryStore::at_work`]): while
/// it stands, the sessions leave what they hand to this thread, which
/// applies it where it finds the turn free before a read of the counts
/// ([`AtWork::apply`]) and at each issue. Dropped, it takes the turn once
/// more: what was handed behind its last take is applied before the
/// thread waits.
pub struct AtWork(Arc<MemoryStore>);

impl AtWork {
    /// Applies what is handed, before a decision reads its rows' counts,
    /// where the turn is free. A turn that is held was taken by a thread
    /// that applied what was handed before it: the decision reads on and
    /// its issue takes the turn.
    pub fn apply(&self) {
        if let Some(mut turn) = self.0.file.try_take() {
            self.0.apply_handed(&mut turn.behind);
        }
    }
}

impl Drop for AtWork {
    /// A thread that leaves by an unwind looks at nothing: what it left
    /// is the next take's, and a tell that unwound is not asked again
    /// under it.
    fn drop(&mut self) {
        self.0.file.at_work.0.fetch_sub(1, SeqCst);
        if !thread::panicking() {
            // The leaver's half of the hand-over ([`MemoryStore::settle`]).
            fence(SeqCst);
            self.0.settle();
        }
    }
}

impl fmt::Debug for AtWork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AtWork").finish_non_exhaustive()
    }
}

/// A stream's hand at the leases' file ([`MemoryStore::handover`]): what
/// its frames say of its gateway's leases goes through it
/// ([`MemoryStore::hand`]).
#[derive(Debug)]
pub struct Hand {
    gateway: GatewayId,
    ring: SyncSender<Handed>,
}

impl Hand {
    /// The gateway whose leases the hand's frames name.
    pub fn gateway(&self) -> GatewayId {
        self.gateway
    }
}

impl MemoryStore {
    /// The frames the ring of the leases' file holds: past them a hand
    /// takes the turn and applies before it writes its own.
    pub const HANDOVER: usize = 1_024;

    /// The file a lease's id names: the one file today.
    fn file_of(&self, _lease: LeaseId) -> &File {
        &self.file
    }

    /// Gives the store the controller's word to the engine: what tells it
    /// of a frame applied to a lease on file. Given once: the first stays.
    #[cfg(any(test, feature = "controller"))]
    pub(crate) fn tell_through(&self, teller: Teller) {
        self.file.teller.get_or_init(|| teller);
    }

    /// Tells the engine what `said` did to the lease `record` stood as,
    /// issued by `by`.
    pub(super) fn tell(&self, record: &LeaseRecord, said: Said, by: Option<Decider>) {
        if let Some(tell) = self.file.teller.get() {
            tell(self, record, said, by);
        }
    }

    /// Counts the calling thread at work at the leases' file until the
    /// answer is dropped: a decider's, from a job at hand to an empty
    /// queue.
    pub fn at_work(self: &Arc<Self>) -> AtWork {
        self.file.at_work.0.fetch_add(1, SeqCst);
        AtWork(Arc::clone(self))
    }

    /// The turn, with everything handed applied.
    pub(super) fn turn(&self) -> Turn<'_> {
        let mut turn = self.file.take();
        self.apply_handed(&mut turn.behind);
        turn
    }

    /// Applies what waits where the caller found no thread at work. A
    /// hand writes its frame and then reads who is at work; a thread that
    /// leaves work says so and then comes here. Each puts a fence between
    /// its write and its read, so one of the two sees the other whatever
    /// the ring's own orderings are: the frame is applied by the thread
    /// at work or by its hand.
    fn settle(&self) {
        drop(self.turn());
    }

    /// Opens a stream's hand at the leases' file for `gateway`'s frames.
    pub fn handover(&self, gateway: GatewayId) -> Hand {
        Hand {
            gateway,
            ring: self.file.hands.clone(),
        }
    }

    /// A stream's end: what it handed is applied before its session
    /// leaves. A hand dropped without this call loses nothing: its frames
    /// are the next take's.
    pub fn hand_back(&self, hand: Hand) {
        drop(hand);
        self.settle();
    }

    /// Hands the leases' file what a frame said of `lease`: a session's
    /// own hand, and the door of a host that ends its leases with no
    /// stream. The frame is written into the file's ring and applied by
    /// whoever holds the turn next: a thread at work at the file, or this
    /// caller where none is. A full ring makes its caller take the turn
    /// and apply, so no frame is dropped and the call's wait is the
    /// turn's.
    pub fn hand(&self, hand: &Hand, lease: LeaseId, said: Said) {
        let file = self.file_of(lease);
        let mut frame = (hand.gateway, lease, said);
        loop {
            match hand.ring.try_send(frame) {
                Ok(()) => break,
                Err(TrySendError::Full(back)) => {
                    frame = back;
                    drop(self.turn());
                }
                // The hand is another store's, whose file is gone: no
                // lease is on file there.
                Err(TrySendError::Disconnected(_)) => return,
            }
        }
        // The hand's half of the hand-over ([`Self::settle`]).
        fence(SeqCst);
        if file.at_work.0.load(SeqCst) == 0 {
            self.settle();
        }
    }

    /// Reads the ring and applies what was handed, under the turn, in the
    /// order it was handed. A tell that unwinds leaves the rest in the
    /// ring.
    fn apply_handed(&self, behind: &mut Behind) {
        let Behind { leases, ring } = behind;
        while let Ok((gateway, lease, said)) = ring.try_recv() {
            self.apply_frame(leases, (gateway, lease), said);
        }
    }

    /// Applies what one frame said of the lease under `key`: the record's
    /// read, the door's change and the word to the engine. A lease not on
    /// file is no error and tells nothing: its dispatch deadline ended it,
    /// or its gateway decided that request itself. Neither does a second
    /// dispatch of one lease.
    fn apply_frame(&self, leases: &mut Leases, key: Key, said: Said) {
        let found = match said {
            Said::Dispatched(dispatch, at) => {
                let found = leases.records.get(&key).map(Held::told);
                leases.dispatched(&self.world, &key, dispatch, at);
                found.filter(|(record, _)| record.state == LeaseState::Issued)
            }
            Said::Completed(outcome) => {
                let found = leases.records.get(&key).map(Held::told);
                leases.completed(&key, &outcome);
                found
            }
            Said::Released(..) => leases.ended(&key).as_ref().map(Held::told),
        };
        if let Some((record, by)) = found {
            self.tell(&record, said, by);
        }
    }
}
