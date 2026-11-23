//! `--entry place`: a cell's requests asked through the one call a host
//! makes for a request ([`PolicyEngine::place`]), instead of the engine's
//! entry.
//!
//! It is asked for a cell whose request is plain and carries no key. The
//! caller states the model, its list and a prompt of the size the plain
//! request states, and says the dispatch and the end of the placement it is
//! answered. It lends no load snapshot: the engine reads its own. The
//! result line names the entry asked (`entry=`).
//!
//! [`PolicyEngine::place`]: smg_policy_controller::engine::PolicyEngine::place

use std::sync::atomic::{AtomicBool, Ordering};

use smg_policy_controller::{
    engine::{Identity, PlaceRequest, Prompt, WorkerLists},
    tracking::{Outcome, Placement, Side, TransportClass},
};

use super::{ruled, Host, Run, MODEL};

/// The prompt a request by the one call states: 1,024 token ids, the size
/// the plain request states.
static PROMPT: [u32; 1_024] = [0; 1_024];

/// The entry `value` names: `decide_at`, the engine's entry, `place`, or
/// `place` with a rule ([`ruled::ENTRY`]).
pub(super) fn named(value: &str) -> String {
    let known = matches!(value, "decide_at" | "place" | ruled::ENTRY);
    assert!(known, "no entry {value}");
    value.to_owned()
}

/// Whether this run's requests go through the one call: what the result
/// line's word says. A run another file measures asks the engine's entry.
static ASKED: AtomicBool = AtomicBool::new(false);

/// Whether `run` asks the one call of `host`'s cell. The call here states a
/// model, a list and a prompt's size: a cell whose request carries a key is
/// not asked through it, and the run stops on one.
pub(super) fn asked(run: &Run, host: &Host) -> bool {
    if run.entry != "place" && run.entry != ruled::ENTRY {
        return false;
    }
    let keyed = (host.request)(0).facts.session_key != Identity::NONE;
    assert!(
        host.plain.is_some() && !keyed,
        "the place entry takes a plain cell whose request carries no key"
    );
    ASKED.store(true, Ordering::Relaxed);
    true
}

/// One request by the one call: the model, the list and a prompt of the
/// plain request's size, and the dispatch and the end said of a placement
/// that owes them. The answer stays where the call wrote it: a placement
/// is not moved. The call answers a worker and no place of the list, so
/// the place this says is the first. With `rule` the request states a
/// route's rule ([`ruled::admits`]).
pub(super) fn one(host: &Host, rule: bool) -> Option<u32> {
    let request = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&PROMPT),
            ..Prompt::NONE
        },
        admits: ruled::admits(rule),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(&host.members))
    };
    let mut answer = host.engine.place(&request);
    let placement = answer.as_mut().ok()?;
    if placement.owes_events() {
        said(placement);
    }
    Some(0)
}

/// The dispatch and the end a host says of a placement, in a function of
/// its own as the engine's entry has one for its guard.
#[inline(never)]
fn said(placement: &mut Placement<'_>) {
    placement.sent(Side::Single);
    let end = Outcome {
        side: Side::Single,
        served: placement.id(),
        status: 200,
        class: TransportClass::Ok,
        ttft_micros: 0,
        tokens_in: 1_024,
        tokens_out: 0,
        duration_micros: 0,
    };
    placement.ended(Side::Single, end);
}

/// The word this file adds at the end of a run's result line: the entry
/// the run's requests went through.
pub(super) fn word() -> &'static str {
    if ASKED.load(Ordering::Relaxed) {
        ruled::word().unwrap_or(" entry=place")
    } else {
        " entry=decide_at"
    }
}
