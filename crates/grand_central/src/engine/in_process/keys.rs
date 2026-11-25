//! A keyed request's decision over a gateway's own list, by the rows its
//! key names.
//!
//! The engine decides a keyed request of the single plan over rows read one
//! at a time ([`Engine::decide_keyed`]): its key's pin is one row, and a
//! ring's walk is the few its key reaches. Here those rows are the records
//! of a gateway's own list, each written when the decision asks about its
//! worker ([`PoolRecords::read`]).

use std::sync::Arc;

use smg_worker::identity::basic::Worker;

use crate::engine::{
    decision::Decision,
    framework::{Engine, PoolRows},
    records::{Live, PoolRecords},
    view::{Candidate, Constraints, RequestFacts, WorkerId},
};

/// One list of a gateway's worker objects as a keyed decision reads it: the
/// list's records, one worker's written at a time.
struct ListRows<'a> {
    records: &'a PoolRecords,
    workers: &'a [Arc<dyn Worker>],
    live: &'a Live<'a>,
}

impl PoolRows for ListRows<'_> {
    fn len(&self) -> usize {
        self.records.len()
    }

    fn read(&self, place: u32) -> Option<Candidate> {
        self.records.read(self.workers, self.live, place)
    }

    fn place_of(&self, id: WorkerId) -> Option<u32> {
        self.records.place_of(id)
    }

    fn generation(&self) -> u64 {
        self.records.generation()
    }
}

/// A keyed request's decision over `workers`, the list `records` were made
/// from, by the rows its key names: its key's pin under the binding's
/// sessions rule, or the workers it reaches on the ring its caller lent
/// ([`RequestFacts::ring`]). `None` where the key decides nothing by those
/// rows: the caller then asks the whole decision ([`records::decide`]) with
/// the same `facts`.
///
/// The decision's instant is written into `facts` here
/// ([`RequestFacts::now_micros`]), the one `live` was made at.
///
/// Marked for inlining across the crate line: the single decision that
/// calls it is compiled for its host's maker, in the host's crate. Without
/// the mark the call stays, 33 instructions a decision.
///
/// [`records::decide`]: crate::engine::records::decide
#[inline]
pub(super) fn decide(
    engine: &Engine,
    records: &PoolRecords,
    workers: &[Arc<dyn Worker>],
    live: &Live<'_>,
    facts: &mut RequestFacts<'_>,
    cx: &Constraints<'_>,
) -> Option<Decision> {
    facts.now_micros = live.now_micros();
    let rows = ListRows {
        records,
        workers,
        live,
    };
    engine.decide_keyed_among(facts, &rows, cx, records.regular())
}

#[cfg(test)]
mod tests;
