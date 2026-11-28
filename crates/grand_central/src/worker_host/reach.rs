//! Who can dial each worker of one list.
//!
//! A worker a controller registers states two facts that no request
//! changes: the wires it speaks, and whether every gateway dials it or one
//! gateway alone holds its host-local socket
//! ([`WorkerFacts::wire_modes`](crate::store::WorkerFacts::wire_modes),
//! [`WorkerFacts::reachable_from`](crate::store::WorkerFacts::reachable_from)).
//! A worker object speaks one wire of the stated set, so the object alone
//! cannot say which gateways dial its worker.
//!
//! [`ListReach`] holds the two facts for one list of worker objects, in
//! the list's order, with what holds for the list as a whole. It is made
//! once a list, and read by each decision over the list for the places its
//! asking gateway may take. A holder cuts one with each list it lends, by
//! the pass that cuts the list ([`Objects::reach`](super::Objects::reach)).
//!
//! **A list's reach is as of its generation.** The wires a worker states do
//! not move within its life: a registration that states others starts
//! another life (`WorkerFacts::compare_to_standing`), and a holder lays
//! another life's object, which the registry publishes. Who dials a worker
//! moves within its life. The holder lays a new object at each move of a
//! row, so the registry publishes a generation and the list's reach is
//! made again.

use super::remainder::RowRemainder;
use crate::{engine::view::WireModes, store::Reachability};

/// One worker's reach: the wires it states, and who dials it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Reach {
    pub(crate) wires: WireModes,
    pub(crate) from: Reachability,
}

impl Reach {
    /// The reach a worker's row states, read where the holder keeps what
    /// the row's object does not hold as stated.
    pub(crate) const fn of(row: &RowRemainder) -> Self {
        Self {
            wires: row.wire_modes,
            from: row.reachable_from,
        }
    }
}

/// How many sets of wires a gateway can state: one for each value the
/// wire set's bits take. [`ListReach`] keeps one bit of a byte for each.
const WIRE_SETS: u8 = WireModes::from_bits(u8::MAX).bits() + 1;
const _: () = assert!(WIRE_SETS <= 8, "one bit of a byte for each set of wires");

/// The reach of every worker of one list, in the list's order, with two
/// facts of the whole list that answer most decisions at two tests
/// ([`Self::whole`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ListReach {
    workers: Box<[Reach]>,
    /// Whether a worker of the list is host-local: one gateway alone dials
    /// it, so another gateway's places leave it out.
    locals: bool,
    /// For each set of wires a gateway can state, by its bits, whether
    /// every worker of the list shares a wire with it.
    dialed_whole: u8,
}

impl ListReach {
    /// The reach of a list whose workers state `workers`, in the list's
    /// order. Its cost grows with the list, once a list.
    pub(crate) fn new(workers: impl IntoIterator<Item = Reach>) -> Self {
        let workers: Box<[Reach]> = workers.into_iter().collect();
        let locals = workers
            .iter()
            .any(|worker| worker.from != Reachability::Any);
        let dialed_whole = (0..WIRE_SETS)
            .filter(|&stated| {
                let wires = WireModes::from_bits(stated);
                workers.iter().all(|worker| worker.wires.intersects(wires))
            })
            .fold(0, |whole, stated| whole | (1 << stated));
        Self {
            workers,
            locals,
            dialed_whole,
        }
    }

    /// Whether a gateway that states `wires` may take every place of the
    /// list, whichever gateway it is: every gateway dials every worker,
    /// and each worker shares a wire with it. A gateway that states no
    /// wires is held to none.
    #[cfg_attr(
        all(not(test), not(feature = "controller")),
        expect(
            dead_code,
            reason = "a controller's deciding thread reads it, and no other build has one"
        )
    )]
    pub(crate) fn whole(&self, wires: Option<WireModes>) -> bool {
        !self.locals && wires.is_none_or(|wires| self.dialed_whole & (1 << wires.bits()) != 0)
    }

    /// Each worker's reach, by its place in the list.
    #[cfg_attr(
        all(not(test), not(feature = "controller")),
        expect(
            dead_code,
            reason = "a controller's deciding thread reads it, and no other build has one"
        )
    )]
    pub(crate) fn workers(&self) -> &[Reach] {
        &self.workers
    }
}

#[cfg(test)]
#[path = "../../tests/unit/worker_host/reach/tests.rs"]
mod tests;
