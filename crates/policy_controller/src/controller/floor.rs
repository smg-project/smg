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
//! ([`crate::elector::TERM_CEILING`]). So the floors one process says
//! move the lease up a budget of terms and no more
//! ([`FLOORS_SPENT_MAX`]): past it a floor above the term is refused as
//! before and says nothing.

use std::sync::OnceLock;

use parking_lot::Mutex;

use super::election::Door;
use crate::elector::term_said;

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

/// What became of a believed floor above the term ([`Floors::say`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Told {
    /// Said to the election.
    Said,
    /// Not said: one as high was said, the election does not run yet, or
    /// the election refuses its term.
    Kept,
    /// Not said: the budget has too few terms left for it. `first` at the
    /// first floor it could not hold.
    Spent { first: bool },
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
}

/// The floors said to the election. The election keeps a term it was told,
/// so a floor is said once: the highest one said is remembered, and only a
/// higher one is said after it.
#[derive(Default)]
pub(crate) struct Floors {
    /// The election's door, set once the election runs. A floor that comes
    /// before it is remembered nowhere: the gateway's next frame says it.
    door: OnceLock<Door>,
    /// What was said. Locked by a floor above the term alone: a frame at
    /// or under the term never comes here.
    words: Mutex<Words>,
}

impl Floors {
    /// The election runs: `door` is its door. A second call changes
    /// nothing.
    pub(crate) fn open(&self, door: Door) {
        let _ = self.door.set(door);
    }

    /// Says `floor`, a believed floor above `term`, unless one as high was
    /// said, the election does not run yet or the budget cannot hold it.
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
        let moved = floor - words.said.max(term) + 1;
        if words.spent + moved > FLOORS_SPENT_MAX {
            let first = !words.out;
            words.out = true;
            return Told::Spent { first };
        }
        // The election refuses a term that leaves none to take, as the
        // floor's judge did before it came here: nothing is said or spent.
        if door.saw(floor).is_err() {
            return Told::Kept;
        }
        words.spent += moved;
        words.said = floor;
        Told::Said
    }

    /// The terms spent so far.
    #[cfg(test)]
    pub(crate) fn spent(&self) -> u64 {
        self.words.lock().spent
    }
}

#[cfg(test)]
mod tests {
    use super::{unbelieved, Floors, Told, FLOORS_SPENT_MAX, FLOOR_BELIEVED};
    use crate::{controller::election::Said, elector::TERM_CEILING};

    #[test]
    fn a_floor_is_believed_up_to_its_range_above_the_term_and_under_the_last_term() {
        assert_eq!(unbelieved(2, 1), None);
        assert_eq!(unbelieved(1 + FLOOR_BELIEVED, 1), None);
        let far = unbelieved(2 + FLOOR_BELIEVED, 1).expect("a floor past the range");
        assert!(far.starts_with("term_floor is 65538: "), "{far}");
        // Inside the range, at the last term: the election could take no
        // term above it.
        let term = TERM_CEILING - 3;
        assert_eq!(unbelieved(TERM_CEILING - 2, term), None);
        let last = unbelieved(TERM_CEILING - 1, term).expect("the last term");
        assert!(last.starts_with("term_floor is 16777215: "), "{last}");
    }

    #[test]
    fn a_floor_is_said_once_and_only_a_higher_one_after_it() {
        let floors = Floors::default();
        // Before the election runs nothing is said and nothing remembered.
        assert_eq!(floors.say(7, 1), Told::Kept);
        let (door, said) = Said::door();
        floors.open(door);
        assert_eq!(floors.say(7, 1), Told::Said);
        assert_eq!(floors.say(7, 1), Told::Kept);
        assert_eq!(floors.say(5, 1), Told::Kept);
        // The lower floor did not take the higher one's place.
        assert_eq!(floors.say(6, 1), Told::Kept);
        assert_eq!(floors.say(9, 1), Told::Said);
        assert_eq!(said.terms(), [7, 9]);
        // A second door changes nothing: the first one hears.
        let (other, unheard) = Said::door();
        floors.open(other);
        assert_eq!(floors.say(10, 1), Told::Said);
        assert_eq!(said.terms(), [10]);
        assert!(unheard.terms().is_empty());
    }

    #[test]
    fn the_floors_said_spend_a_budget_of_terms_counted_from_the_highest_said() {
        let floors = Floors::default();
        let (door, said) = Said::door();
        floors.open(door);
        // Two floors under one term: the second moves the lease up from
        // the first, not from the term. Each spends the take's term too.
        assert_eq!(floors.say(11, 1), Told::Said);
        assert_eq!(floors.say(16, 1), Told::Said);
        assert_eq!(floors.spent(), 17);
        // A floor that is not said spends nothing.
        assert_eq!(floors.say(16, 1), Told::Kept);
        assert_eq!(floors.spent(), 17);

        // One term past the budget is refused its word, the first time by
        // name, and spends nothing. The budget whole is said.
        let term = 100;
        let rest = FLOORS_SPENT_MAX - 17;
        // The floor whose take lands on the budget's last term.
        let whole = term + rest - 1;
        let past = whole + 1;
        assert_eq!(floors.say(past, term), Told::Spent { first: true });
        assert_eq!(floors.say(past, term), Told::Spent { first: false });
        assert_eq!(floors.spent(), 17);
        assert_eq!(floors.say(whole, term), Told::Said);
        assert_eq!(floors.spent(), FLOORS_SPENT_MAX);
        // Spent: a floor one above the term the election took says nothing.
        assert_eq!(floors.say(past + 1, past), Told::Spent { first: false });
        assert_eq!(said.terms(), [11, 16, whole]);
    }

    #[test]
    fn a_floor_the_election_refuses_is_neither_said_nor_spent() {
        let floors = Floors::default();
        let (door, said) = Said::door();
        floors.open(door);
        // The session's judge refuses this floor before it comes here; the
        // election's own refusal counts nothing.
        assert_eq!(floors.say(TERM_CEILING - 1, TERM_CEILING - 3), Told::Kept);
        assert_eq!(floors.spent(), 0);
        assert!(said.terms().is_empty());
        // The floor under it is said: the refused one was not remembered.
        assert_eq!(floors.say(TERM_CEILING - 2, TERM_CEILING - 3), Told::Said);
        assert_eq!(said.terms(), [TERM_CEILING - 2]);
    }

    #[test]
    fn words_of_one_term_each_move_the_lease_the_budget_and_no_further() {
        let floors = Floors::default();
        let (door, _said) = Said::door();
        floors.open(door);
        // Each word is one above the term and its take one above the word:
        // two terms of the lease a word.
        let start = 1;
        let mut term = start;
        let mut words = 0;
        // No more words than terms: a budget that never refuses ends here.
        while words <= FLOORS_SPENT_MAX && floors.say(term + 1, term) == Told::Said {
            words += 1;
            term += 2;
        }
        assert_eq!(words, FLOORS_SPENT_MAX / 2);
        assert_eq!(term - start, FLOORS_SPENT_MAX);
    }
}
