//! A gateway's standing vetoes, written into the records of a decision
//! that shows them to a plugin.
//!
//! A host that decides for many gateways hands one list's records to each
//! of them. It states the asking gateway's standing vetoes beside the
//! records: the places they hold back ([`Constraints::stalled`]) and those
//! of them no rule lifts ([`Constraints::held`]). The crate's own stages
//! read a record's bit and the mask beside it.
//!
//! A plugin is shown each record as the record's own fields, and no mask.
//! The engine declines a profile that shows its rows to a plugin over rows
//! that carry a mask beside them, under the stage name `written-rows`. So
//! for such a profile the vetoes are written into the records
//! ([`written`]), in the form of a row written for its decision: the
//! stalled bit on each masked record, and the breaker bit beside it where
//! the veto holds ([`Health`](crate::engine::view::Health)). The decision
//! is then put to constraints that state no mask.
//!
//! The records are those the decision wrote for itself, in its thread's
//! own buffer ([`PoolRecords::write`](super::PoolRecords::write)): no
//! other decision reads them.

use crate::engine::{
    framework::Engine,
    view::{Candidate, Constraints, Phase},
};

/// The constraints a decision of `phase` for `model` is put to over
/// `rows`, every record of its list written for it, where its host states
/// a standing veto beside them. Where the decision's profile shows its
/// rows to a plugin ([`Engine::reads_for`]), the vetoes are written into
/// `rows` and the constraints state no mask. That word carries a pair's
/// side for a decode or an encode decision of a model that also pairs: a
/// decode alone under a binding ahead of the pair's has its vetoes written
/// where the pair's decode side shows its rows. Under any other profile
/// `rows` stand as written and the constraints are `cx`'s own.
///
/// A record a hold stands on carries both bits, as the record of a worker
/// whose breaker refuses execution does. A place past the list's end
/// names no record.
///
/// Cold and out of line: a decision under no veto asks one test of its
/// constraints and never comes here.
#[cold]
#[inline(never)]
pub(crate) fn written<'a>(
    engine: &Engine,
    model: &str,
    phase: Phase,
    rows: &mut [Candidate],
    cx: &Constraints<'a>,
) -> Constraints<'a> {
    let (_, shown, _) = engine.reads_for(model, phase);
    if !shown {
        return *cx;
    }
    for &place in cx.stalled {
        if let Some(row) = rows.get_mut(place as usize) {
            row.health.stalled = true;
        }
    }
    for &place in cx.held {
        if let Some(row) = rows.get_mut(place as usize) {
            row.health.stalled = true;
            row.health.breaker_open = true;
        }
    }
    Constraints {
        stalled: &[],
        held: &[],
        ..*cx
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/engine/records/vetoes/tests.rs"]
mod tests;
