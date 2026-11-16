//! The decision open on a thread: what the plugins of one deciding host
//! share while it is open, whatever their kind. A decision is one unit to
//! everything it opens: it is numbered once, its budget is seeded once, and
//! every call of every plugin and side it opens draws the same remainder.

use std::{
    cell::{RefCell, RefMut},
    sync::Arc,
};

use thread_local::ThreadLocal;

/// The decisions open on a deciding host's threads, one a thread, each with
/// its number and what is left of its budget. Whoever hosts the engine makes
/// one of these and builds every plugin kind's runtime over it. A plugin is
/// loaded over a runtime, so the plugins of every runtime built over the
/// same one share the open decision: its first open on a thread numbers it
/// and seeds its budget, and each call of each of them draws that
/// remainder, whichever engine build opened it.
///
/// Runtimes built over two of these share nothing: a decision that opens
/// plugins of both is two decisions to them, each with a remainder of its
/// own.
///
/// The premise: a decision is opened and closed on one thread, with no
/// other decision between. The engine keeps it: its decide is a plain
/// function. A host's own stage must keep it too. One that decides again
/// on that thread starts no second decision: its opens join the open one
/// (one count, one remainder), and a plugin bound in both is asked over
/// the outer decision's view and closed by the inner one's close.
#[derive(Debug, Default)]
pub struct OpenDecisions {
    /// By `thread_local`'s thread id. That crate recycles the id of a thread
    /// that exited, and a cell at rest has no open, so an adopter's first
    /// open starts a decision of its own under the next number.
    threads: ThreadLocal<RefCell<OpenDecision>>,
}

impl OpenDecisions {
    /// No decision open on any thread.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// The decision open on this thread.
    pub(crate) fn on_this_thread(&self) -> RefMut<'_, OpenDecision> {
        self.threads.get_or_default().borrow_mut()
    }
}

/// The decision open on a thread and its budget, in microseconds of the
/// request's: numbered and seeded by its first open, drawn down by every
/// call of every plugin and side the decision opens, forgotten by its last
/// close.
#[derive(Debug, Default)]
pub(crate) struct OpenDecision {
    pub(crate) remaining_micros: u64,
    /// The request's budget as seeded: a plugin's own spend reads against it.
    pub(crate) seed_micros: u64,
    /// The decision's number on this thread, never zero once one opened: a
    /// plugin keys its part in the decision by it ([`Part`]).
    number: u64,
    opens: u32,
}

impl OpenDecision {
    /// One more open of the decision on this thread (a plugin, a side, a
    /// binding): the first numbers the decision and seeds its budget from
    /// the request's. Returns the decision's number, the one way to learn
    /// it: a kind keeps what its open returned.
    pub(crate) fn open(&mut self, budget_micros: u32) -> u64 {
        self.opens += 1;
        if self.opens == 1 {
            self.remaining_micros = u64::from(budget_micros);
            self.seed_micros = u64::from(budget_micros);
            self.number = self.number.wrapping_add(1).max(1);
        }
        self.number
    }

    /// One close of the decision on this thread: after its last, the next
    /// open starts another decision.
    pub(crate) fn close(&mut self) {
        self.opens = self.opens.saturating_sub(1);
    }
}

/// One plugin's part in the decision open on a thread: the decision's
/// number, the plugin's opens of it still to close (each side of a pair,
/// each binding), and whether the plugin already failed it. The decision is
/// the breaker's unit: once a plugin failed one, its later opens of it
/// bypass, so a threshold of N failures is N decisions on a pair as on a
/// single placement. A kind keeps one a plugin and thread, beside the
/// plugin's own time in the decision.
///
/// A part is keyed by the number alone: equal numbers are one decision to
/// it. A plugin is bound into every engine build that names it, an earlier
/// build still deciding while a later one starts, so the numbers a plugin
/// sees come from one issuer as wide as that sharing: the
/// [`OpenDecisions`] its runtime was built over.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Part {
    number: u64,
    opens: u32,
    failed: bool,
}

impl Part {
    /// Whether `number` is the decision this part is of. A kind asks before
    /// an open, to start its own time over with the part.
    pub(crate) fn is_of(&self, number: u64) -> bool {
        self.number == number
    }

    /// One of the plugin's opens of decision `number`; says whether the
    /// plugin already failed that decision, the one question an open asks. A
    /// number the part does not hold starts it over.
    pub(crate) fn open(&mut self, number: u64) -> bool {
        if self.number != number {
            *self = Self {
                number,
                opens: 0,
                failed: false,
            };
        }
        self.opens = self.opens.saturating_add(1);
        self.failed
    }

    /// The plugin failed: the decision it has open is marked. Outside a
    /// decision (a lifecycle event) nothing is marked.
    pub(crate) fn fail(&mut self) {
        if self.opens != 0 {
            self.failed = true;
        }
    }

    /// Whether the plugin already failed decision `number`; true until the
    /// thread's next decision opens the plugin.
    pub(crate) fn failed_in(&self, number: u64) -> bool {
        self.number == number && self.failed
    }

    /// One of the plugin's closes of decision `number`: `true` at its last,
    /// where a kind takes the decision's sample. A number the part does not
    /// hold closes nothing.
    pub(crate) fn close(&mut self, number: u64) -> bool {
        if self.number != number {
            return false;
        }
        self.opens = self.opens.saturating_sub(1);
        self.opens == 0
    }
}

#[cfg(test)]
mod tests {
    use super::{OpenDecisions, Part};

    #[test]
    fn the_first_open_numbers_and_seeds_and_later_opens_draw_the_same_remainder() {
        let open = OpenDecisions::new();
        let mut decision = open.on_this_thread();
        let first = decision.open(900);
        assert_ne!(first, 0, "a decision's number is never zero");
        assert_eq!(
            (decision.remaining_micros, decision.seed_micros),
            (900, 900)
        );
        decision.remaining_micros -= 400;
        // The other side of a pair, or another plugin: the same decision,
        // whatever budget its facts carry, and the remainder as drawn.
        assert_eq!(decision.open(700), first);
        assert_eq!(
            (decision.remaining_micros, decision.seed_micros),
            (500, 900)
        );
        decision.close();
        assert_eq!(decision.open(700), first, "an open is still to close");
        decision.close();
        decision.close();
        let second = decision.open(300);
        assert_ne!(second, first, "the last close ended the decision");
        assert_eq!(
            (decision.remaining_micros, decision.seed_micros),
            (300, 300)
        );
    }

    #[test]
    fn each_thread_holds_a_decision_of_its_own() {
        let open = OpenDecisions::new();
        let here = open.on_this_thread().open(900);
        open.on_this_thread().remaining_micros = 1;
        let there = std::thread::scope(|scope| {
            let other = scope.spawn(|| {
                let mut decision = open.on_this_thread();
                let number = decision.open(500);
                (number, decision.remaining_micros)
            });
            other.join()
        });
        assert!(
            matches!(there, Ok((_, 500))),
            "another thread's open seeds its own remainder: {there:?}"
        );
        assert_eq!(
            open.on_this_thread().open(700),
            here,
            "this thread's decision is the one it opened"
        );
        assert_eq!(open.on_this_thread().remaining_micros, 1);
    }

    #[test]
    fn a_part_marks_the_decision_its_plugin_failed_until_the_next_one_opens_it() {
        let mut part = Part::default();
        assert!(!part.open(7), "a fresh decision was not failed");
        part.fail();
        assert!(part.failed_in(7));
        assert!(part.open(7), "the other side finds the decision failed");
        assert!(!part.close(7), "one open is still to close");
        assert!(part.close(7), "the last close");
        assert!(part.failed_in(7), "read at the close's cleanup too");
        assert!(!part.open(8), "the next decision starts the part over");
        assert!(part.is_of(8) && !part.is_of(7));
        assert!(!part.failed_in(7));
        part.fail();
        assert!(
            part.failed_in(8) && !part.failed_in(7),
            "the mark is the open decision's"
        );
        assert!(part.close(8));
    }

    #[test]
    fn a_failure_outside_a_decision_marks_nothing() {
        let mut part = Part::default();
        assert!(!part.open(3));
        assert!(part.close(3));
        part.fail();
        assert!(!part.failed_in(3), "a lifecycle failure after the close");
        assert!(
            !part.close(4),
            "a close of a decision the part does not hold"
        );
    }

    #[test]
    fn equal_numbers_are_one_decision_to_a_part() {
        // Why one issuer hands out every number a plugin sees: a part cannot
        // tell two decisions with one number apart.
        let mut part = Part::default();
        assert!(!part.open(1));
        part.fail();
        assert!(part.close(1));
        assert!(part.is_of(1));
        assert!(
            part.open(1),
            "the same number reads as the decision already failed"
        );
    }
}
