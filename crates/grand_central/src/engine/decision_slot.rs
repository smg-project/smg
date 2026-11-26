//! A decision slot's name: how a stage kind that keeps state a slot finds it.
//!
//! A host decides on its deciders, one decision at a time on each. The
//! served controller gives a model's requests one decider. Whoever decides
//! for a decider holds its turn: the decider's own thread, or another thread
//! the host lets take the turn. What a stage kind keeps a decider (a
//! WebAssembly plugin's instance) is that decider's, behind its turn,
//! whichever thread holds it. A stage's calls and the lifecycle's carry no
//! handle of the host, so the holder of a turn says whose it is:
//! [`DecisionSlot::enter`] names the decider the calling thread acts as until
//! the guard drops, and a kind reads [`DecisionSlot::acting`].
//!
//! A host enters where it takes a decider's turn: no stage and no plugin
//! kind does. The served controller enters at two
//! places. A decider's own thread enters for each request it decides. And
//! whoever applies a lease's event enters as the decider whose file of
//! leases the lease is on, under that file's turn: the decider's thread
//! itself, or another thread while that decider waits for a request, never
//! beside it. So the calls made under one decider's name run one at a
//! time. A host that names none decides as before: a thread that names no
//! decider is a decider of its own.
//!
//! ```
//! use smg_grand_central::engine::DecisionSlot;
//!
//! // The holder of the third decider's turn: a decision made here, and a
//! // lease's event told here, reach what that decider keeps.
//! let turn = DecisionSlot::new(2).enter();
//! assert_eq!(DecisionSlot::acting(), Some(DecisionSlot::new(2)));
//! drop(turn);
//! assert_eq!(DecisionSlot::acting(), None);
//! ```

use std::{
    cell::{Cell, RefCell},
    marker::PhantomData,
};

thread_local! {
    /// The decider the thread acts as: the name of its newest live guard.
    static ACTING: Cell<Option<DecisionSlot>> = const { Cell::new(None) };
    /// The thread's live guards, oldest first, each by its number and name.
    static LIVE: RefCell<Vec<(u64, DecisionSlot)>> = const { RefCell::new(Vec::new()) };
}

/// One of a host's decision slots, by the number the host gave it: a byte,
/// so a host names [`DecisionSlot::COUNT`] at most.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DecisionSlot(u8);

impl DecisionSlot {
    /// How many deciders a host can name.
    pub const COUNT: usize = 1 << u8::BITS;

    /// The decider a host numbers `number`.
    #[must_use]
    pub const fn new(number: u8) -> Self {
        Self(number)
    }

    /// The number the host gave it.
    #[must_use]
    pub const fn number(self) -> u8 {
        self.0
    }

    /// The calling thread acts as this decider until the guard drops: what
    /// the holder of the decider's turn says before it decides or tells a
    /// lease's event, for as long as it holds the turn and no longer. A
    /// thread that enters again acts as the later name. Guards drop in any
    /// order: the thread acts as the name of its newest guard still alive,
    /// so a guard kept in a field and renewed by assignment keeps its name,
    /// and the thread acts as none once its last guard is gone. At its
    /// thread's end, once the thread's list of guards is gone, a guard
    /// entered by another thread-local's destructor names the thread until
    /// a guard drops there, its own or another's: with no list no older
    /// guard is known, and the thread then acts as none.
    #[must_use = "the thread acts as the decider only while the guard lives"]
    pub fn enter(self) -> DecisionSlotGuard {
        // A guard entered once the list is gone joins none: its number is
        // not read again.
        let number = LIVE
            .try_with(|live| {
                let mut live = live.borrow_mut();
                let number = live.last().map_or(0, |(newest, _)| newest + 1);
                live.push((number, self));
                number
            })
            .unwrap_or(0);
        ACTING.set(Some(self));
        DecisionSlotGuard {
            number,
            thread: PhantomData,
        }
    }

    /// The decider the calling thread acts as, where its host named one.
    ///
    /// A stage kind that keeps state a decider reads it here. A door in
    /// process seats a request by its model only under a generation that
    /// holds a stage opening a decision's scope
    /// ([`Engine::keeps_slot_state`](crate::engine::framework::Engine::keeps_slot_state)).
    /// Under any other it seats the request by its caller's thread, and state a
    /// stage keeps a decider then splits by thread, with no error.
    #[must_use]
    pub fn acting() -> Option<Self> {
        ACTING.get()
    }
}

/// A thread's hold on a decider's name ([`DecisionSlot::enter`]). It is its
/// thread's and is never sent to another, so no name is held across a wait
/// that may resume elsewhere:
///
/// ```compile_fail
/// fn sent<T: Send>(_: T) {}
/// sent(smg_grand_central::engine::DecisionSlot::new(0).enter());
/// ```
#[derive(Debug)]
pub struct DecisionSlotGuard {
    /// The guard's number among its thread's live guards: one more than the
    /// newest one's at its entry, so it grows by one a renewal for as long
    /// as any guard lives. No two guards of the list share a number: a `u64`
    /// does not come round. A guard entered past the list takes zero, which
    /// nothing reads.
    number: u64,
    /// A raw pointer is not `Send`, so the guard is not.
    thread: PhantomData<*const ()>,
}

impl Drop for DecisionSlotGuard {
    fn drop(&mut self) {
        // The guard leaves the thread's list wherever it stands in it, and
        // the newest one left names the thread. Where the list is gone (a
        // guard kept in another thread-local, or entered by one's
        // destructor, at its thread's end) no older guard is known, and the
        // thread acts as none.
        let left = LIVE.try_with(|live| {
            let mut live = live.borrow_mut();
            if let Some(at) = live.iter().rposition(|(number, _)| *number == self.number) {
                live.remove(at);
            }
            live.last().map(|(_, name)| *name)
        });
        ACTING.set(left.ok().flatten());
    }
}

#[cfg(test)]
#[path = "../../tests/unit/engine/decision_slot/tests.rs"]
mod tests;
