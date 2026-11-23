//! A request held to places of its pool.
//!
//! A request's host can hold it to a part of its pool: the one worker a
//! request names, the workers that meet its route's rule. A retry's
//! history is not stated so: it is the workers the request tried. The
//! places are the request's own fact
//! ([`Constraints::places`]) and no veto of a worker. A document chooses
//! how a worker's health is weighed; it does not choose which workers a
//! request can be sent to. So the rule is the engine's own, whatever a
//! profile lists, and no health bit of a record says a narrowing.
//!
//! Every way a decision opens a set or reads a row holds the rule, in one
//! of three forms:
//!
//! - **The filter half opens with the places** ([`open`]). Every staged
//!   pass of a single request goes through it: the whole pass's stages,
//!   the sessions rule and the placing assignment, the walk under a lifted
//!   veto, each pass of the all-vetoed rule and the ranks. So the rule for
//!   a pool whose every worker is vetoed steers among the places.
//! - **What opens a set of its own, or reads a row by its place, steps
//!   aside** ([`stated`]): the one pass that asks a profile's filters as
//!   one test of a row, the draw, the draw over a pool's engines, a
//!   scorer's own walk over them, and the keyed decision, which reads a
//!   key's pin and the workers of a ring's walk by their places. Each
//!   says nothing of such a request, inside itself, and the staged pass
//!   decides, as it does under the draw's floor and behind a draw that
//!   keeps nobody.
//! - **A look at a side's rows that opens no set reads the places' rows**
//!   ([`rows`]): a pair's list of open decodes, the decodes whose classes
//!   its all-vetoed rule opens, and what its prefill side's shed says of
//!   the prefills.
//!
//! **A pin outside the places.** A sessions rule reads a pin whose worker
//! is outside the places as it reads a pin whose worker a filter refuses.
//! A pin inside the places is followed, as over the whole list: one at
//! its rule's cap is placed again. Where no pin is inside them, the
//! rule's recovery places the key among the places and claims the worker
//! picked, and a dispatch to that worker confirms the claim. A key's one
//! pin stands in the table, the claim behind it under `original` and
//! ahead of it under `latest`. Of two pins one leaves for the claim: the
//! first under `original`, the second under `latest`.
//!
//! Not held to the places: the pick of a plain request, which reads no
//! constraint. So a request that states places is not plain.
//!
//! The places are those of the one list a set is opened over. A pair's
//! host states each side's places beside its two lists, and the pair's
//! entry is handed one set of constraints a side: each side opens with its
//! own, and a decode outside its side's places partners no prefill. One
//! set of places cannot hold a pair's two lists: a pair's request that
//! states places of its own is declined by the engine over worker lists,
//! and the served controller states none.
//!
//! **Places a request already tried.** A request held to places whose
//! every worker it tried is handed one of them again, as over a list of
//! those workers alone: the filter of tried workers is soft and steps
//! aside where it would leave nobody.

use crate::engine::view::{Candidate, Constraints};

/// Whether the request's host holds it to places of its pool: the one test
/// of a pass that opens a set of its own or reads a row by its place, which
/// then steps aside.
#[inline]
pub(super) const fn stated(cx: &Constraints<'_>) -> bool {
    cx.places.is_some()
}

/// The rows the request may take, in the pool's order: every row, or the
/// rows at the places it is held to. The form of a look at a side's rows
/// that opens no set: a pair's list of open decodes, the decodes whose
/// classes its all-vetoed rule opens, and what its prefill side's shed
/// says of the prefills.
pub(super) fn rows<'a>(
    candidates: &'a [Candidate],
    cx: &Constraints<'a>,
) -> impl Iterator<Item = &'a Candidate> {
    let (every, held): (&[Candidate], &[u32]) = match cx.places {
        None => (candidates, &[]),
        Some(places) => (&[], places),
    };
    every.iter().chain(
        held.iter()
            .filter_map(move |&at| candidates.get(at as usize)),
    )
}

/// Opens a decision's set over a pool of `rows` rows, into `kept`: every
/// row, or the places the request is held to and no other row.
#[inline]
pub(super) fn open(kept: &mut Vec<u32>, rows: usize, cx: &Constraints<'_>) {
    match cx.places {
        None => kept.extend(0..rows as u32),
        Some(places) => among(kept, rows, places),
    }
}

/// The places below the pool's end, into `kept`, in the order its host
/// states them: ascending, each once ([`Constraints::places`]). Out of
/// line: a set opened over every row keeps the code it had.
#[cold]
#[inline(never)]
fn among(kept: &mut Vec<u32>, rows: usize, places: &[u32]) {
    debug_assert!(
        places.windows(2).all(|pair| pair[0] < pair[1]),
        "the places are ascending, each once"
    );
    kept.extend(places.iter().copied().filter(|&at| (at as usize) < rows));
}

#[cfg(test)]
mod tests;
