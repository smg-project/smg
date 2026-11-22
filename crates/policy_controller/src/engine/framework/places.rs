//! A request held to places of its pool.
//!
//! A request's host can hold it to a part of its pool: the workers a retry
//! may still take, the one worker a request names, the workers that meet
//! its route's rule. The places are the request's own fact
//! ([`Constraints::places`]) and no veto of a worker. A document chooses
//! how a worker's health is weighed; it does not choose which workers a
//! request can be sent to. So the rule is the engine's own, whatever a
//! profile lists, and no health bit of a record says a narrowing.
//!
//! Two ways a decision opens a set hold the rule:
//!
//! - **The filter half opens with the places** ([`open`], [`hold`]). Every
//!   staged pass goes through it: the whole pass's stages, the sessions
//!   rule and the placing assignment, the walk under a lifted veto, each
//!   pass of the all-vetoed rule and the ranks. So the rule for a pool
//!   whose every worker is vetoed steers among the places.
//! - **The one pass that asks a profile's filters as one test of a row
//!   steps aside** ([`held`]): it says nothing of such a request, and the
//!   staged pass decides.
//!
//! Not held to the places yet: a draw (over a pool's rows, over its
//! engines, a scorer's own walk over them) and a row read by its place in
//! the keyed decision (a key's pin, the members of a ring's walk). No host
//! states places until each of them asks.
//!
//! The places are those of the one list a single request is decided over.
//! A pair's sides and an encode item open through the same filter half, so
//! each of their lists would be held to the same places: a host of a pair
//! states none.

use crate::engine::view::Constraints;

/// Whether the request's host holds it to places of its pool: the one test
/// of a pass that opens a set of its own, which then steps aside.
#[inline]
pub(super) const fn held(cx: &Constraints<'_>) -> bool {
    cx.places.is_some()
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

/// The places below the pool's end, into `kept`. Out of line: a set opened
/// over every row keeps the code it had.
#[cold]
#[inline(never)]
fn among(kept: &mut Vec<u32>, rows: usize, places: &[u32]) {
    kept.extend(places.iter().copied().filter(|&at| (at as usize) < rows));
}

/// Keeps of `kept`, the rows a set opened with by a rule of its own, those
/// among the places the request is held to. Nothing where it is held to
/// none: one test.
#[inline]
pub(super) fn hold(kept: &mut Vec<u32>, cx: &Constraints<'_>) {
    if let Some(places) = cx.places {
        kept.retain(|at| places.binary_search(at).is_ok());
    }
}

#[cfg(test)]
mod tests;
