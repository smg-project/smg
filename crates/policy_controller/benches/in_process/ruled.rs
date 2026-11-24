//! `--entry place_ruled`: the one call ([`super::place`]) asked with a
//! route's own rule stated, one that admits every worker. What stating a
//! rule costs a request is this entry's count less the `place` entry's.

use std::sync::atomic::{AtomicBool, Ordering};

use smg_policy_controller::engine::WorkerRule;

use super::Run;

/// The word of `--entry` that asks the one call with a route's rule stated.
pub(super) const ENTRY: &str = "place_ruled";

/// Whether this run's requests state the rule: what the result line's word
/// says.
static STATED: AtomicBool = AtomicBool::new(false);

/// Whether `run` asks the one call with the rule stated.
pub(super) fn stated(run: &Run) -> bool {
    let stated = run.entry == ENTRY;
    STATED.store(stated, Ordering::Relaxed);
    stated
}

/// The route's rule a request states: one that admits every worker where
/// its run states it, and none otherwise.
pub(super) fn admits(stated: bool) -> Option<WorkerRule<'static>> {
    let every: WorkerRule<'static> = &|_| true;
    stated.then_some(every)
}

/// The result line's word for a run that states the rule.
pub(super) fn word() -> Option<&'static str> {
    STATED
        .load(Ordering::Relaxed)
        .then_some(" entry=place_ruled")
}
