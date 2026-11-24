//! `--entry place`: a cell's requests asked through the one call a host
//! makes for a request ([`PolicyEngine::place`]), instead of the engine's
//! entry.
//!
//! It is asked for a cell whose request is plain and carries no key, and
//! for an encode request's cell, whose own file states its three lists
//! ([`asked_with_lists`]). For a plain cell the caller states the model,
//! its list and a prompt of the size the plain request states, and says
//! the dispatch and the end of the placement it is answered. It lends no
//! load snapshot: the engine reads its own. The result line names the
//! entry asked (`entry=`).
//!
//! Every request of a plain cell states its prompt's entry in a token tree
//! ([`Prompt::entry`]), as a host that lends one states it for each
//! request. A cell whose document reads no tree never has it asked: its
//! count is what the stated field costs unread. A cell whose callers hold
//! conversations states each turn's own ([`turn`]).
//!
//! [`PolicyEngine::place`]: smg_policy_controller::engine::PolicyEngine::place

use std::{
    borrow::Cow,
    sync::atomic::{AtomicBool, Ordering},
};

use smg_policy_controller::{
    engine::{Identity, PlaceRequest, Prompt, TokenEntry, TokenKey, WorkerLists},
    policies::consistent_hashing::LentRing,
    tracking::{Outcome, Placement, Side, TransportClass},
};

use super::{conversations, ruled, worded, Host, Run, MODEL, RING};

/// The prompt a request by the one call states: 1,024 token ids, the size
/// the plain request states.
pub(super) static PROMPT: [u32; 1_024] = [0; 1_024];

/// The plain prompt's entry in a token tree: its token ids, under no cache
/// partition.
struct Whole;

impl TokenEntry for Whole {
    fn token_key(&self, _: usize) -> Option<TokenKey<'_>> {
        Some(TokenKey {
            tokens: Cow::Borrowed(&PROMPT),
            marker: 0,
        })
    }
}

/// The entry `value` names: `decide_at`, the engine's entry, `place`,
/// `place` with a rule ([`ruled::ENTRY`]), or `place` with a prompt of text
/// ([`worded::ENTRY`]).
pub(super) fn named(value: &str) -> String {
    let known = matches!(value, "decide_at" | "place" | ruled::ENTRY | worded::ENTRY);
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
    if !matches!(&*run.entry, "place" | ruled::ENTRY | worded::ENTRY) {
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

/// The ring `host` lends with each request by the one call: the run's
/// ring under a cell whose document reads one, as a host that keeps rings
/// lends its own with such a request, and none under any other document.
pub(super) fn lent(host: &Host) -> Option<&'static dyn LentRing> {
    let reads = host.engine.spec().needs().ring;
    let ring = RING.get().filter(|_| reads)?;
    Some(ring)
}

/// One request by the one call: the model, the list and a prompt of the
/// plain request's size, and the dispatch and the end said of a placement
/// that owes them. The answer stays where the call wrote it: a placement
/// is not moved. The call answers a worker and no place of the list, so
/// the place this says is the first. With `rule` the request states a
/// route's rule ([`ruled::admits`]), and it lends `ring` ([`lent`]).
pub(super) fn one(host: &Host, rule: bool, ring: Option<&'static dyn LentRing>) -> Option<u32> {
    let request = PlaceRequest {
        prompt: Prompt {
            tokens: Some(&PROMPT),
            entry: Some(&Whole),
            ..Prompt::NONE
        },
        admits: ruled::admits(rule),
        ring,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(host.list.lent()))
    };
    let mut answer = host.engine.place(&request);
    let placement = answer.as_mut().ok()?;
    if placement.owes_events() {
        said(placement);
    }
    Some(0)
}

/// Whether this run's requests go through the one call, for a cell whose
/// callers make their own turns, and whether they state a route's rule
/// then ([`ruled::ENTRY`]).
pub(super) fn asks() -> Option<bool> {
    let asked = ASKED.load(Ordering::Relaxed);
    asked.then(|| ruled::word().is_some())
}

/// One turn of a conversation by the one call: the model, the list, the
/// turn's token ids and its entry in the tree the cell's host lends. The
/// engine hashes the prompt itself here, where the engine's entry is handed
/// the hashes. Says the worker the turn was placed on, by its id's low
/// half: the call answers a worker and no place of the list. With `rule`
/// the turn states a route's rule ([`ruled::admits`]), which a decision
/// made inside the tree asks ahead of the tree's write.
pub(super) fn turn(host: &Host, prompt: &conversations::Turn<'_>, rule: bool) -> Option<u32> {
    let request = PlaceRequest {
        prompt: Prompt {
            tokens: Some(prompt.tokens),
            entry: Some(prompt),
            ..Prompt::NONE
        },
        admits: ruled::admits(rule),
        ..PlaceRequest::of(MODEL, WorkerLists::Single(host.list.lent()))
    };
    let mut answer = host.engine.place(&request);
    let placement = answer.as_mut().ok()?;
    let worker = placement.id().0 as u32;
    if placement.owes_events() {
        said(placement);
    }
    Some(worker)
}

/// The dispatch and the end a host says of a placement, in a function of
/// its own as the engine's entry has one for its guard.
#[inline(never)]
pub(super) fn said(placement: &mut Placement<'_>) {
    placement.sent(Side::Single);
    let mut end =
        Outcome::reported(200, TransportClass::Ok).with_served(Side::Single, placement.id());
    end.tokens_in = 1_024;
    placement.ended(Side::Single, end);
}

/// The word this file adds at the end of a run's result line: the entry
/// the run's requests went through.
pub(super) fn word() -> &'static str {
    if ASKED.load(Ordering::Relaxed) {
        ruled::word()
            .or_else(worded::word)
            .unwrap_or(" entry=place")
    } else {
        " entry=decide_at"
    }
}

/// Whether `run` asks the one call of a cell that hands lists of its own,
/// which its own file runs: an encode request's. Such a cell states no
/// route's rule, and the run stops on one. The result line's word says the
/// entry asked.
pub(super) fn asked_with_lists(run: &Run) -> bool {
    assert!(run.entry != ruled::ENTRY, "such a cell states no rule");
    let asked = run.entry == "place";
    ASKED.store(asked, Ordering::Relaxed);
    asked
}
