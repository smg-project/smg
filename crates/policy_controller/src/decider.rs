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

use std::{cell::Cell, marker::PhantomData};

thread_local! {
    /// The decider the thread acts as.
    static ACTING: Cell<Option<Decider>> = const { Cell::new(None) };
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
    /// thread that enters again acts as the later name, and as the earlier
    /// one once the later guard drops. The order is the holder's to keep:
    /// guards drop in the reverse of the order they were made in, as locals
    /// do. A guard dropped while a later one lives leaves the thread under
    /// the wrong name from there on, and a debug build panics at that drop.
    #[must_use = "the thread acts as the decider only while the guard lives"]
    pub fn enter(self) -> Acting {
        Acting {
            name: self,
            before: ACTING.replace(Some(self)),
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
    /// The name this guard set.
    name: Decider,
    /// What the thread acted as before.
    before: Option<Decider>,
    /// A raw pointer is not `Send`, so the guard is not.
    thread: PhantomData<*const ()>,
}

impl Drop for Acting {
    fn drop(&mut self) {
        // The newest guard drops first. A thread that unwinds is past the
        // check: a second panic there would end the process.
        debug_assert!(
            std::thread::panicking() || ACTING.get() == Some(self.name),
            "a decider's guard dropped while a later one lives: acting {:?}, the guard's {:?}",
            ACTING.get(),
            self.name
        );
        ACTING.set(self.before);
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

    /// The order is the holder's to keep, and a debug build says where it
    /// was not kept. On a thread of its own: the name it leaves is wrong.
    #[test]
    #[cfg(debug_assertions)]
    fn a_guard_dropped_while_a_later_one_lives_panics_in_a_debug_build() {
        let dropped = std::thread::spawn(|| {
            let earlier = Decider::new(1).enter();
            let _later = Decider::new(2).enter();
            drop(earlier);
        })
        .join();
        let panic = dropped.expect_err("the drop out of its order panics");
        let said = panic.downcast_ref::<String>().map_or("", String::as_str);
        assert!(said.contains("dropped while a later one lives"), "{said}");
    }
}
