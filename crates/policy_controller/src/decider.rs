//! A decider's name: how a stage kind that keeps state a decider finds it.
//!
//! A host decides a model's requests one at a time on one of its deciders,
//! and whoever decides for a decider holds its turn: the decider's own
//! thread, or another thread the host lets take the turn. What a stage kind
//! keeps a decider (a WebAssembly plugin's instance) is that decider's,
//! behind its turn, whichever thread holds it. A stage's calls and the
//! lifecycle's carry no handle of the host, so the holder of a turn says
//! whose it is: [`Decider::enter`] names the decider the calling thread acts
//! as until the guard drops, and a kind reads [`Decider::acting`].
//!
//! One place of a host enters, where it takes a turn: no stage, no plugin
//! kind and no door of a store does. This is the one naming of a decider in
//! the crate. A host that names none decides as before: a thread that names
//! no decider is a decider of its own.
//!
//! ```
//! use smg_policy_controller::decider::Decider;
//!
//! // The holder of the third decider's turn: a decision made here, and a
//! // lease's event told here, reach what that decider keeps.
//! let turn = Decider::new(2).enter();
//! assert_eq!(Decider::acting(), Some(Decider::new(2)));
//! drop(turn);
//! assert_eq!(Decider::acting(), None);
//! ```

use std::{
    cell::{Cell, RefCell},
    marker::PhantomData,
};

thread_local! {
    /// The decider the thread acts as: the name of its newest live guard.
    static ACTING: Cell<Option<Decider>> = const { Cell::new(None) };
    /// The thread's live guards, oldest first, each by its number and name.
    static LIVE: RefCell<Vec<(u32, Decider)>> = const { RefCell::new(Vec::new()) };
}

/// One of a host's deciders, by the number the host gave it: a byte, so a
/// host names [`Decider::COUNT`] at most.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Decider(u8);

impl Decider {
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
    /// and the thread acts as none once its last guard is gone.
    #[must_use = "the thread acts as the decider only while the guard lives"]
    pub fn enter(self) -> Acting {
        let number = LIVE.with_borrow_mut(|live| {
            let number = live.last().map_or(0, |(newest, _)| newest + 1);
            live.push((number, self));
            number
        });
        ACTING.set(Some(self));
        Acting {
            number,
            thread: PhantomData,
        }
    }

    /// The decider the calling thread acts as, where its host named one.
    #[must_use]
    pub fn acting() -> Option<Self> {
        ACTING.get()
    }
}

/// A thread's hold on a decider's name ([`Decider::enter`]). It is its
/// thread's and is never sent to another, so no name is held across a wait
/// that may resume elsewhere:
///
/// ```compile_fail
/// fn sent<T: Send>(_: T) {}
/// sent(smg_policy_controller::decider::Decider::new(0).enter());
/// ```
#[derive(Debug)]
pub struct Acting {
    /// The guard's number among its thread's live guards.
    number: u32,
    /// A raw pointer is not `Send`, so the guard is not.
    thread: PhantomData<*const ()>,
}

impl Drop for Acting {
    fn drop(&mut self) {
        // The guard leaves the thread's list wherever it stands in it, and
        // the newest one left names the thread. A guard that outlives the
        // list (one kept in another thread-local, at its thread's end) has
        // nothing to end.
        let _ = LIVE.try_with(|live| {
            let mut live = live.borrow_mut();
            if let Some(at) = live.iter().rposition(|(number, _)| *number == self.number) {
                live.remove(at);
            }
            ACTING.set(live.last().map(|(_, name)| *name));
        });
    }
}

#[cfg(test)]
mod tests {
    use super::Decider;

    #[test]
    fn a_thread_acts_as_the_decider_it_entered_until_the_guard_drops() {
        assert_eq!(Decider::acting(), None);
        let outer = Decider::new(3).enter();
        assert_eq!(Decider::acting(), Some(Decider::new(3)));
        {
            let _inner = Decider::new(7).enter();
            assert_eq!(Decider::acting().map(Decider::number), Some(7));
        }
        assert_eq!(
            Decider::acting(),
            Some(Decider::new(3)),
            "the earlier name again"
        );
        drop(outer);
        assert_eq!(Decider::acting(), None);
    }

    #[test]
    fn a_name_is_its_threads_and_no_others() {
        let _held = Decider::new(1).enter();
        let other = std::thread::spawn(Decider::acting)
            .join()
            .expect("the other thread");
        assert_eq!(other, None);
        assert_eq!(Decider::acting(), Some(Decider::new(1)));
    }

    #[test]
    fn a_guard_dropped_while_a_later_one_lives_leaves_the_later_name() {
        let earlier = Decider::new(1).enter();
        let later = Decider::new(2).enter();
        drop(earlier);
        assert_eq!(Decider::acting(), Some(Decider::new(2)));
        let newest = Decider::new(3).enter();
        drop(later);
        assert_eq!(Decider::acting(), Some(Decider::new(3)));
        drop(newest);
        assert_eq!(Decider::acting(), None, "no guard is left");
    }

    /// A guard kept in a field and renewed by assignment: the new guard
    /// enters before the old one drops, and the name holds at each renewal.
    #[test]
    fn a_guard_renewed_by_assignment_keeps_its_name() {
        let mut held = Decider::new(4).enter();
        for _ in 0..3 {
            // What an assignment does: the new guard enters, the old drops.
            drop(std::mem::replace(&mut held, Decider::new(4).enter()));
            assert_eq!(Decider::acting(), Some(Decider::new(4)));
        }
        drop(held);
        assert_eq!(Decider::acting(), None);
    }
}
