//! `--entry place_ruled`: the one call ([`super::place`]) asked with a
//! route's own rule stated, one that admits every worker. What stating a
//! rule costs a request is this entry's count less the `place` entry's.
//!
//! `--entry place_held`: the one call asked with a rule that takes the
//! last worker of each list and no other. A request's first decision names
//! workers the rule refuses, the rule is asked of each list, and the call
//! decides again among the workers it takes. What that costs a request is
//! this entry's count less the `place` entry's. A first decision that
//! names the rule's workers is taken at once, and asks the rule of no
//! list.
//!
//! A single request's cell, a pair's and an encode request's take both
//! entries. An encode request's rule is its pair's: it is asked of the
//! prefills and of the decodes, and of no encoder.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    OnceLock,
};

use smg_grand_central::engine::WorkerRule;
use smg_worker::identity::basic::Worker;

use super::Run;

/// The word of `--entry` that asks the one call with a rule that admits
/// every worker.
pub(super) const ENTRY: &str = "place_ruled";

/// The word of `--entry` that asks the one call with a rule that takes the
/// last worker of each list alone.
pub(super) const HELD: &str = "place_held";

/// Whether `entry` asks the one call with a route's rule stated.
pub(super) fn names(entry: &str) -> bool {
    matches!(entry, ENTRY | HELD)
}

/// Whether this run's requests state a rule: what the result line's word
/// says.
static STATED: AtomicBool = AtomicBool::new(false);

/// How the address of each list's last worker ends, under a run whose rule
/// takes that worker alone ([`HELD`]): a cell numbers the workers of each
/// list its rule is asked of alike. Unset under every other run.
static LAST: OnceLock<String> = OnceLock::new();

/// Whether `run` asks the one call with a rule stated.
pub(super) fn stated(run: &Run) -> bool {
    let stated = names(&run.entry);
    if run.entry == HELD {
        LAST.get_or_init(|| format!("-{:04}:8000", run.workers - 1));
    }
    STATED.store(stated, Ordering::Relaxed);
    stated
}

/// The route's rule a request states: the run's where its run states one,
/// and none otherwise. It admits every worker, or under [`HELD`] the last
/// worker of each list alone.
pub(super) fn admits(stated: bool) -> Option<WorkerRule<'static>> {
    let every: WorkerRule<'static> = &|_| true;
    let last: WorkerRule<'static> = &|worker: &dyn Worker| {
        LAST.get()
            .is_some_and(|ends| worker.url().ends_with(ends.as_str()))
    };
    stated.then(|| if LAST.get().is_some() { last } else { every })
}

/// The result line's word for a run that states a rule.
pub(super) fn word() -> Option<&'static str> {
    let held = LAST.get().is_some();
    STATED.load(Ordering::Relaxed).then_some(if held {
        " entry=place_held"
    } else {
        " entry=place_ruled"
    })
}
