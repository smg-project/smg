//! The decision open on a thread: what the plugins of one deciding host
//! share while it is open, whatever their kind. A decision is one unit to
//! everything it opens: it is numbered once, its budget is seeded once, and
//! every call of every plugin and side it opens draws the same remainder.

#[cfg(feature = "controller")]
use std::time::{Duration, Instant};
use std::{
    cell::{RefCell, RefMut},
    sync::Arc,
};

use thread_local::ThreadLocal;

use super::breaker::Latency;
#[cfg(feature = "controller")]
use super::config::FrameBudget;
use crate::engine::framework::{self, DecisionScope};

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

/// What is left of a decision's budget on its thread, and the budget as
/// seeded, in microseconds of the request's. A budget begins at a
/// decision's first open, which seeds it ([`OpenDecision::open`]), or
/// outside every decision under a bound of its own ([`Budget::alone`]).
/// Every call of every plugin and side the decision opens asks here what
/// it may spend, and draws the remainder down here and nowhere else
/// ([`Budget::draw`]): the remainder only falls, and never below zero.
/// That rule is of the two fields: a whole budget can still be assigned
/// through [`OpenDecision::budget`].
#[derive(Debug, Default)]
pub(crate) struct Budget {
    remaining_micros: u64,
    seed_micros: u64,
}

impl Budget {
    /// A budget of `micros` that no decision seeded: what a call outside
    /// every decision runs under (a lifecycle event, a destructor), a bound
    /// of its own. Its seed is zero. So [`Self::own_bound`] answers yes for
    /// every call under it, and such a call's frame names no decision: its
    /// run-out is its plugin's own and is never laid on a decision's budget.
    #[cfg(feature = "wasm")]
    pub(crate) fn alone(micros: u32) -> Self {
        Self {
            remaining_micros: u64::from(micros),
            seed_micros: 0,
        }
    }

    /// A budget seeded with `micros`, for a case of a kind that closes a
    /// decision under one.
    #[cfg(all(test, feature = "controller"))]
    pub(crate) fn seeded(micros: u32) -> Self {
        let mut budget = Self::default();
        budget.seed(micros);
        budget
    }

    /// Seeds the budget from the request's: a decision's first open.
    fn seed(&mut self, micros: u32) {
        self.remaining_micros = u64::from(micros);
        self.seed_micros = u64::from(micros);
    }

    /// Draws `micros` from the remainder, which never falls below zero: the
    /// one write of a field after the budget began, seeded by a decision's
    /// first open or made alone ([`Budget::alone`]).
    pub(crate) fn draw(&mut self, micros: u64) {
        self.remaining_micros = self.remaining_micros.saturating_sub(micros);
    }

    /// Whether nothing is left: an open or a call that asks now gets no
    /// frame.
    pub(crate) fn is_spent(&self) -> bool {
        self.remaining_micros == 0
    }

    /// What a call of the WebAssembly kind may spend: its fuel at
    /// `fuel_per_micro` under a frame of `cap` fuel, and its time on the
    /// clock under a frame of `frame_micros`. Each is the frame, or what is
    /// left where that is less. The two frames are the call's own and need
    /// not agree. No fuel says no call. A kind bounds a call here, and
    /// reads no remainder to bound one.
    #[cfg(feature = "wasm")]
    pub(crate) fn grant_call(
        &self,
        cap: u64,
        frame_micros: u32,
        fuel_per_micro: u64,
    ) -> (u64, u32) {
        (
            cap.min(self.fuel(fuel_per_micro)),
            frame_micros.min(self.remaining_micros_u32()),
        )
    }

    /// What a frame of the gRPC kind is sent with, for a plugin whose
    /// frame budget is `frame`. First what the frame may spend: the
    /// plugin's own frame, or what is left where that is less. Then what
    /// is left, which the frame tells its plugin: zero says no frame.
    #[cfg(feature = "controller")]
    pub(crate) fn grant_frame(&self, frame: FrameBudget) -> (u32, u32) {
        let left = self.remaining_micros_u32();
        (frame.frame_micros(left), left)
    }

    /// How long a wait for a frame's answer may last. The wait began at
    /// `started` and the frame's own deadline is `deadline`: the wait ends
    /// at the deadline, or where what is left runs out if that is sooner.
    #[cfg(feature = "controller")]
    pub(crate) fn grant_wait(&self, started: Instant, deadline: Instant) -> Instant {
        deadline.min(started + Duration::from_micros(self.remaining_micros))
    }

    /// What a frame with no answer was given. Its wait began
    /// `before_micros` after a frame of `frame_micros` was sent: the frame
    /// as sent, or the time ahead of the wait and what is left where that
    /// is less. A kind asks behind a missed wait, ahead of the frame's
    /// draw: an answered frame never asks.
    #[cfg(feature = "controller")]
    pub(crate) fn grant_missed(&self, frame_micros: u32, before_micros: u64) -> u64 {
        u64::from(frame_micros).min(before_micros.saturating_add(self.remaining_micros))
    }

    /// What is left, in the width a frame and the wire carry it. A budget
    /// is seeded from a `u32` and only falls, so nothing is cut off.
    fn remaining_micros_u32(&self) -> u32 {
        u32::try_from(self.remaining_micros).unwrap_or(u32::MAX)
    }

    /// What is left, as fuel at `fuel_per_micro`.
    #[cfg(feature = "wasm")]
    fn fuel(&self, fuel_per_micro: u64) -> u64 {
        self.remaining_micros.saturating_mul(fuel_per_micro)
    }

    /// Whether the bound a call ran out under was its plugin's own: the call
    /// had its whole frame, or the plugin alone consumed the budget (its own
    /// time in the decision before the call, `spent_micros`, and the call's
    /// grant in full, `granted_micros`, reach the budget as seeded). Short of
    /// both, the shared remainder clamped the call (other plugins drew it, or
    /// the host held a call of this one): its run-out is the decision's
    /// budget spent and no failure. Every kind asks here whose bound a
    /// run-out was, and nowhere else.
    ///
    /// The WebAssembly kind reads `whole_frame` against its frame under the
    /// request's budget and the gRPC kind against its plugin's own frame,
    /// with the same verdicts. A grant of the whole budget under a wider
    /// frame reaches the budget by itself.
    pub(crate) fn own_bound(
        &self,
        whole_frame: bool,
        spent_micros: u64,
        granted_micros: u64,
    ) -> bool {
        whole_frame || reaches_the_budget(spent_micros, granted_micros, self.seed_micros)
    }

    /// The frame a decision's latency sample is taken over, for a plugin
    /// that was given `frames_micros` of frames in the decision: those
    /// frames, at most the budget as seeded ([`Latency::frame`]). A kind
    /// asks at its last close of the decision.
    pub(crate) fn sample_frame(&self, frames_micros: u64) -> u32 {
        Latency::frame(self.seed_micros, frames_micros)
    }
}

/// The decision open on a thread and its budget: numbered and seeded by its
/// first open, drawn down by every call of every plugin and side the
/// decision opens, forgotten by its last close.
#[derive(Debug, Default)]
pub(crate) struct OpenDecision {
    budget: Budget,
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
            self.budget.seed(budget_micros);
            self.number = self.number.wrapping_add(1).max(1);
        }
        self.number
    }

    /// The decision's budget: the one way to it.
    pub(crate) fn budget(&mut self) -> &mut Budget {
        &mut self.budget
    }

    /// One close of the decision on this thread: after its last, the next
    /// open starts another decision.
    pub(crate) fn close(&mut self) {
        self.opens = self.opens.saturating_sub(1);
    }
}

/// Whether a plugin alone consumed its decision's budget: its own time in
/// the decision before a call, `spent_micros`, and the call's grant in
/// full, `granted_micros`, reach the budget as seeded. Exact: a microsecond
/// short, some other draw took part of the budget, and a call that runs out
/// there is the decision's budget spent.
fn reaches_the_budget(spent_micros: u64, granted_micros: u64, seed_micros: u64) -> bool {
    spent_micros.saturating_add(granted_micros) >= seed_micros
}

/// An open of a plugin's bound decision that has not returned yet. The
/// engine closes a scope only after its open returned, and a plugin's open
/// marks its phase open and counts the decision before its own steps. So an
/// open that stops half-way takes both back itself, by the walk the close
/// does. Every kind's open runs under one of these.
pub(crate) struct HalfOpen<'a, S: DecisionScope> {
    scope: &'a S,
    returned: bool,
}

impl<'a, S: DecisionScope> HalfOpen<'a, S> {
    /// The guard of `scope`'s open, made before the open's first step.
    pub(crate) fn new(scope: &'a S) -> Self {
        Self {
            scope,
            returned: false,
        }
    }

    /// The open returned: the scope's close is the engine's from here.
    pub(crate) fn returned(&mut self) {
        self.returned = true;
    }

    /// The open is unwinding, and a second panic let out here would end the
    /// process. So the walk runs contained: if it panics too (the host's
    /// clock again, a destructor's host call), it stops there and the open's
    /// panic is the one the host sees. Out of line: an open that returned
    /// pays the guard's one test.
    #[cold]
    #[inline(never)]
    fn undo(&self) {
        framework::contained(|| self.scope.close());
    }
}

impl<S: DecisionScope> Drop for HalfOpen<'_, S> {
    fn drop(&mut self) {
        if !self.returned {
            self.undo();
        }
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
#[path = "../../../tests/unit/plugins/shared/open/tests.rs"]
mod tests;
