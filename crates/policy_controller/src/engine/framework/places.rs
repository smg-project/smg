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
//! - **The filter half opens with the places** ([`open`]). Every staged
//!   pass of a single request goes through it: the whole pass's stages,
//!   the sessions rule and the placing assignment, the walk under a lifted
//!   veto, each pass of the all-vetoed rule and the ranks. So the rule for
//!   a pool whose every worker is vetoed steers among the places.
//! - **The one pass that asks a profile's filters as one test of a row
//!   steps aside** ([`stated`]): it says nothing of such a request, and the
//!   staged pass decides.
//!
//! Not held to the places yet: a draw (over a pool's rows, over its
//! engines, a scorer's own walk over them), a row read by its place in
//! the keyed decision (a key's pin, the members of a ring's walk), and the
//! pick of a plain request, which reads no constraint. So a request that
//! states places is not plain, and until each of the others asks, the
//! engine over worker lists declines it by name.
//!
//! The places are those of the one list a single request is decided over,
//! and one set of places cannot hold a pair's two lists. A prefill side, a
//! decode side under its partnered list and an encode item open through
//! the same filter half and would each be held to them. A decode side a
//! class narrows opens by its classes and reads none. So the engine over
//! worker lists declines a pair that states places, and the served
//! controller states none.
//!
//! **Places a request already tried.** A request held to places whose
//! every worker it tried is handed one of them again, as over a list of
//! those workers alone: the filter of tried workers is soft and steps
//! aside where it would leave nobody.

use crate::engine::view::Constraints;

/// Whether the request's host holds it to places of its pool: the one test
/// of a pass that opens a set of its own, which then steps aside.
#[inline]
pub(super) const fn stated(cx: &Constraints<'_>) -> bool {
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
