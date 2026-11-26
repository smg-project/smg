//! A gateway's term floor above the term this replica leads under
//! (design 5.3). Gateways learn terms from every replica's `Leader` answer
//! and raise their floor. A leader shown a floor above its own term has not
//! noticed that it lost, or leads under a lease that was wiped beside a
//! gateway that remembers. Its session refuses the frame, as before, and
//! says the floor to the election: the leader gives the lease up and the
//! floor counts among the terms its run saw, so the next take is above it.
//! A term never goes back while a gateway that saw it lives.
//!
//! A floor is believed as told, within a range. One further above the term
//! than [`FLOOR_BELIEVED`], or one whose next term the election cannot
//! take, is refused by its field's name and says nothing.
//!
//! A gateway's word is a lever on the terms, and the terms end
//! ([`crate::service::elector::TERM_CEILING`]). So the floors one process says
//! move the lease up a budget of terms and no more
//! ([`FLOORS_SPENT_MAX`]): past it a floor above the term is refused as
//! before and says nothing. A budget is one process's and the lease
//! outlives every process, so no floor moves the lease past a mark either
//! ([`FLOOR_REACH`]): the terms above it are kept for hand-overs, whatever
//! any process said before.

use std::sync::OnceLock;

use parking_lot::Mutex;

use super::election::ElectionNotifier;
use crate::service::elector::{term_said, TERM_CEILING};

/// How far above the term a gateway's floor is believed. A floor counts the
/// hand-overs a fleet lived through and stays far under it. One further
/// above is a fault's (a flipped bit, a gateway moved from another fleet)
/// and costs its own gateway alone. A fleet that lived through more
/// hand-overs and is redeployed whole restarts its gateways.
pub(crate) const FLOOR_BELIEVED: u64 = 1 << 16;

/// Why a floor above `term` is not believed, in the refusal's words; `None`
/// when it is.
pub(crate) fn unbelieved(floor: u64, term: u64) -> Option<String> {
    if floor.saturating_sub(term) > FLOOR_BELIEVED {
        return Some(format!(
            "term_floor is {floor}: a floor is believed up to {FLOOR_BELIEVED} above this controller's term {term}"
        ));
    }
    term_said(Some(floor)).is_err().then(|| {
        format!("term_floor is {floor}: its next term is not below the election's ceiling")
    })
}

/// How many terms the floors one process says move the lease up, in all:
/// sixteen times the believed range, a sixteenth of the terms an election
/// can take. Each believed floor spends terms for good, and words
/// alone could spend them all. Nothing honest comes near the budget: a
/// fault costs a floor or two. Past it a floor above the term moves
/// nothing until the process starts again; the fence still ends a leader
/// that lost.
pub(crate) const FLOORS_SPENT_MAX: u64 = 16 * FLOOR_BELIEVED;

/// The highest term a floor's word moves the lease to, in every process:
/// half the terms an election can take. A budget renews when its process
/// starts again and the lease keeps the term, so the budgets of sixteen
/// process lifetimes would spend every term, and no replica could take the
/// lease after that. The terms above the reach are kept for hand-overs. A
/// floor whose take would land past it is refused as before and moves
/// nothing.
pub(crate) const FLOOR_REACH: u64 = TERM_CEILING / 2;

// A floor under the reach leaves a term to take: the election's door
// refuses no word the floors say.
const _: () = assert!(FLOOR_REACH < TERM_CEILING - 1);

/// What became of a believed floor above the term ([`Floors::say`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Told {
    /// Said to the election.
    Said,
    /// Not said: one as high was said, or the election does not run yet.
    Kept,
    /// Not said: the budget has too few terms left for it. `first` at the
    /// first floor it could not hold.
    Spent { first: bool },
    /// Not said: its take would land past the reach. `first` at the first
    /// such floor.
    Past { first: bool },
}

/// What the floors said so far come to.
#[derive(Default)]
struct Words {
    /// The highest floor said; zero before the first.
    said: u64,
    /// The terms the floors said moved the lease up, in all.
    spent: u64,
    /// Whether a floor was refused its word for the budget.
    out: bool,
    /// Whether a floor was refused its word for the reach.
    past: bool,
}

/// The floors said to the election. The election keeps a term it was told,
/// so a floor is said once: the highest one said is remembered, and only a
/// higher one is said after it.
#[derive(Default)]
pub(crate) struct Floors {
    /// The election's door, set once the election runs. A floor that comes
    /// before it is remembered nowhere: the gateway's next frame says it.
    door: OnceLock<ElectionNotifier>,
    /// What was said. Locked by a floor above the term alone: a frame at
    /// or under the term never comes here.
    words: Mutex<Words>,
}

impl Floors {
    /// The election runs: `door` is its door. A second call changes
    /// nothing.
    pub(crate) fn open(&self, door: ElectionNotifier) {
        let _ = self.door.set(door);
    }

    /// Says `floor`, a believed floor above `term`, unless one as high was
    /// said, the election does not run yet, its take would land past the
    /// reach or the budget cannot hold it.
    /// The lease moves up from the higher of the term and the floor last
    /// said to the next take's term, one above the floor: that many terms
    /// are spent.
    pub(crate) fn say(&self, floor: u64, term: u64) -> Told {
        let Some(door) = self.door.get() else {
            return Told::Kept;
        };
        let mut words = self.words.lock();
        if words.said >= floor {
            return Told::Kept;
        }
        // The take after a floor's word is one above the floor.
        if floor >= FLOOR_REACH {
            let first = !words.past;
            words.past = true;
            return Told::Past { first };
        }
        let moved = floor - words.said.max(term) + 1;
        if words.spent + moved > FLOORS_SPENT_MAX {
            let first = !words.out;
            words.out = true;
            return Told::Spent { first };
        }
        words.spent += moved;
        words.said = floor;
        // Under the reach the door refuses no term.
        let _ = door.saw(floor);
        Told::Said
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/service/controller/floor/tests.rs"]
mod tests;
