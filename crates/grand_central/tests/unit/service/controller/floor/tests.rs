use super::{unbelieved, Floors, Told, FLOORS_SPENT_MAX, FLOOR_BELIEVED, FLOOR_REACH};
use crate::service::{controller::election::Said, elector::TERM_CEILING};

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
fn a_floor_whose_take_would_pass_the_reach_is_refused_its_word() {
    let floors = Floors::default();
    let (door, said) = Said::door();
    floors.open(door);
    // The contract says the reach by its number.
    assert_eq!(FLOOR_REACH, 8_388_608);
    // The take after a floor is one above it: the floor under the reach
    // is the last a word moves the lease for.
    let term = FLOOR_REACH - 2;
    assert_eq!(floors.say(FLOOR_REACH, term), Told::Past { first: true });
    assert_eq!(
        floors.say(FLOOR_REACH + 1, term),
        Told::Past { first: false }
    );
    assert_eq!(floors.spent(), 0);
    assert!(said.terms().is_empty());
    assert_eq!(floors.say(FLOOR_REACH - 1, term), Told::Said);
    assert_eq!(floors.spent(), 2);
    assert_eq!(said.terms(), [FLOOR_REACH - 1]);
    // A leader that hand-overs brought past the reach: no floor moves
    // it, however little was spent.
    assert_eq!(
        floors.say(FLOOR_REACH + 9, FLOOR_REACH + 8),
        Told::Past { first: false }
    );
    assert!(said.terms().is_empty());
}

#[test]
fn the_floors_of_every_process_lifetime_stop_at_the_reach() {
    // The lease keeps the term a process's floors moved it to, and the
    // next process starts with a whole budget.
    let mut term = 1;
    let mut lifetimes = 0;
    loop {
        let floors = Floors::default();
        let (door, _said) = Said::door();
        floors.open(door);
        let started = term;
        // A floor at the top of the believed range a word.
        while floors.say(term + FLOOR_BELIEVED, term) == Told::Said {
            term += FLOOR_BELIEVED + 1;
        }
        if term == started {
            break;
        }
        lifetimes += 1;
        // Sixteen budgets are every term: a mark that held none of
        // these floors back ends the case here.
        assert!(
            lifetimes < TERM_CEILING / FLOORS_SPENT_MAX,
            "the floors of {lifetimes} lifetimes moved the lease to {term}"
        );
    }
    // Never past the reach, and within one word of it.
    assert!(term <= FLOOR_REACH, "{term}");
    assert!(term + FLOOR_BELIEVED >= FLOOR_REACH, "{term}");
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
