//! `--entry place_text`: the one call ([`super::place`]) asked with the
//! request's prompt stated as text, with no token id. What a request of
//! text costs over one of token ids is this entry's count less the `place`
//! entry's: under a document a stage of which reads a request's size, the
//! pass that counts the text's characters is in it.

use std::sync::atomic::{AtomicBool, Ordering};

use smg_policy_controller::{
    engine::{PlaceRequest, Prompt, WorkerLists},
    policies::consistent_hashing::LentRing,
};

use super::{place, Host, Run, MODEL};

/// The word of `--entry` that asks the one call with a prompt of text.
pub(super) const ENTRY: &str = "place_text";

/// The prompt such a request states: 4,096 one-byte characters, the plain
/// request's 1,024 tokens at four bytes a token.
static TEXT: &str = match std::str::from_utf8(&[b't'; 4_096]) {
    Ok(text) => text,
    Err(_) => panic!("the bytes are text"),
};

/// Whether this run's requests state their prompt as text: what the result
/// line's word says.
static STATED: AtomicBool = AtomicBool::new(false);

/// Whether `run` asks the one call with a prompt of text.
pub(super) fn stated(run: &Run) -> bool {
    let stated = run.entry == ENTRY;
    STATED.store(stated, Ordering::Relaxed);
    stated
}

/// One request by the one call with its prompt stated as text, and the
/// dispatch and the end said of a placement that owes them, as the `place`
/// entry says them ([`place::one`]). It lends `ring` as that entry does
/// ([`place::lent`]).
pub(super) fn one(host: &Host, ring: Option<&'static dyn LentRing>) -> Option<u32> {
    let request = PlaceRequest {
        prompt: Prompt {
            text: Some(TEXT),
            ..Prompt::NONE
        },
        ring,
        ..PlaceRequest::of(MODEL, WorkerLists::Single(host.list.lent()))
    };
    let mut answer = host.engine.place(&request);
    let placement = answer.as_mut().ok()?;
    if placement.owes_events() {
        place::said(placement);
    }
    Some(0)
}

/// The result line's word for a run that states its prompt as text.
pub(super) fn word() -> Option<&'static str> {
    STATED
        .load(Ordering::Relaxed)
        .then_some(" entry=place_text")
}
