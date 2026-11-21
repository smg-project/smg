//! A keyed request's decision over a gateway's own list, by the rows its
//! key names.
//!
//! The engine decides a keyed request of the single plan over rows read one
//! at a time ([`Engine::decide_keyed`]): its key's pin is one row, and a
//! ring's walk is the key's owner and the members behind it. Here those rows
//! are the records of a gateway's own list, each written when the decision
//! asks about its worker ([`PoolRecords::read`]), and the ring is the
//! list's own: built at the first decision that reads a ring over the list,
//! kept with the list and never changed.

use std::sync::{Arc, OnceLock};

use smg_worker::identity::basic::Worker;

use crate::{
    engine::{
        decision::Decision,
        framework::{Engine, PoolRows},
        records::{Live, PoolRecords},
        view::{Candidate, Constraints, RequestFacts, WorkerId},
    },
    policies::consistent_hashing::PoolRing,
};

/// One list of a gateway's worker objects as a keyed decision reads it: the
/// list's records, one worker's written at a time, and the list's ring.
struct ListRows<'r, 'a> {
    ring: &'r OnceLock<PoolRing>,
    records: &'a PoolRecords,
    workers: &'a [Arc<dyn Worker>],
    live: &'a Live<'a>,
}

impl<'r> PoolRows<'r> for ListRows<'r, '_> {
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

    fn ring(&self) -> Option<&'r PoolRing> {
        Some(self.ring.get_or_init(|| ring_of(self.records)))
    }
}

/// The ring of the list `records` were made from: each worker by its id and
/// its stable key, at its place in the list. Built once a list, by the
/// first decision that reads it: a decision that asks meanwhile waits for
/// that one build.
fn ring_of(records: &PoolRecords) -> PoolRing {
    PoolRing::new(records.fixed().iter().map(|row| (row.id, row.key)))
}

/// A keyed request's decision over `workers`, the list `records` were made
/// from, by the rows its key names: its key's pin under the binding's
/// sessions rule, or its key's owner on `ring`, the list's own, and the
/// members behind it. `None` where the key decides nothing by those rows:
/// the caller then asks the whole decision ([`records::decide`]) with the
/// same `facts`, which carry the list's ring from here where the binding
/// reads one.
///
/// The decision's instant is written into `facts` here
/// ([`RequestFacts::now_micros`]), the one `live` was made at.
///
/// [`records::decide`]: crate::engine::records::decide
pub(super) fn decide<'r>(
    engine: &Engine,
    ring: &'r OnceLock<PoolRing>,
    records: &PoolRecords,
    workers: &[Arc<dyn Worker>],
    live: &Live<'_>,
    facts: &mut RequestFacts<'r>,
    cx: &Constraints<'_>,
) -> Option<Decision> {
    facts.now_micros = live.now_micros();
    let rows = ListRows {
        ring,
        records,
        workers,
        live,
    };
    engine.decide_keyed(facts, &rows, cx)
}

#[cfg(test)]
mod tests;
