//! A lead lost between two rows. A registration asks the term ahead of
//! each row it writes and each row it removes, and a life a row at a
//! time. A refusal stops the write there, and what was written before
//! stands. The same add under a term that holds lays the rest.

use super::*;

/// A life is asked a row at a time: a turn that may make no second life
/// refuses there, and the row written before it stands.
#[test]
fn a_life_refused_between_two_rows_leaves_the_row_before() {
    struct OneLife<'a>(&'a MemoryStore, std::cell::Cell<bool>);

    impl Turn for OneLife<'_> {
        over_the_table!(0);

        fn life(&self) -> Result<u64, MembershipWriteRefused> {
            if self.1.replace(true) {
                return Err(MembershipWriteRefused::NotLeading(
                    "the term moved".to_owned(),
                ));
            }
            Ok(7)
        }

        fn room(&self, _: &WorkerFacts, _: &Registrant) -> Result<(), MembershipWriteRefused> {
            Ok(())
        }
    }

    let (store, _) = door();
    let two = ranked(2);
    let states: Vec<_> = two
        .iter()
        .map(|facts| move |_: Option<&WorkerFacts>| facts.clone())
        .collect();
    let rows: Vec<_> = two
        .iter()
        .zip(&states)
        .map(|(facts, state)| Stated {
            url: &facts.key,
            state,
        })
        .collect();
    let turn = OneLife(&store, std::cell::Cell::new(false));
    assert_eq!(
        lay(&turn, &Registrant::Control, Precondition::Any, &rows),
        Err(MembershipWriteRefused::NotLeading(
            "the term moved".to_owned()
        ))
    );
    assert!(store.by_key(&two[0].key).is_some());
    assert!(store.by_key(&two[1].key).is_none());
}

/// A turn whose term holds for a number of checks and is lost then.
struct TermFor<'a>(&'a MemoryStore, std::cell::Cell<usize>);

impl Turn for TermFor<'_> {
    over_the_table!(0);

    fn check_term(&self) -> Result<(), MembershipWriteRefused> {
        let Some(left) = self.1.get().checked_sub(1) else {
            return Err(MembershipWriteRefused::NotLeading(
                "the term moved".to_owned(),
            ));
        };
        self.1.set(left);
        Ok(())
    }

    fn life(&self) -> Result<u64, MembershipWriteRefused> {
        Ok(7)
    }

    fn room(&self, _: &WorkerFacts, _: &Registrant) -> Result<(), MembershipWriteRefused> {
        Ok(())
    }
}

fn lay_under(
    turn: &TermFor<'_>,
    facts: &[WorkerFacts],
) -> Result<Vec<MembershipWriteOutcome>, MembershipWriteRefused> {
    lay_as(turn, Precondition::Any, facts)
}

fn lay_as(
    turn: &TermFor<'_>,
    expect: Precondition,
    facts: &[WorkerFacts],
) -> Result<Vec<MembershipWriteOutcome>, MembershipWriteRefused> {
    let states: Vec<_> = facts
        .iter()
        .map(|facts| move |_: Option<&WorkerFacts>| facts.clone())
        .collect();
    let rows: Vec<_> = facts
        .iter()
        .zip(&states)
        .map(|(facts, state)| Stated {
            url: &facts.key,
            state,
        })
        .collect();
    lay(turn, &Registrant::Control, expect, &rows)
}

/// The term is asked ahead of every write, whatever the write makes of
/// its row. A label laid in place and a rank that leaves are refused
/// behind a lost lead, and what was written before it stands.
#[test]
fn a_term_lost_between_two_rows_stops_a_write_in_place_and_a_removal() {
    let (store, door) = door();
    let three = ranked(3);
    find_rows(&door, &Registrant::Control, &three).expect("three ranks registered");
    let lost = Err(MembershipWriteRefused::NotLeading(
        "the term moved".to_owned(),
    ));
    let term = |checks| TermFor(&store, std::cell::Cell::new(checks));

    // Every rank's label changes. The term holds for the first row.
    let mut labelled = three.clone();
    for facts in &mut labelled {
        facts.labels.insert("tier".to_owned(), "gold".to_owned());
    }
    assert_eq!(lay_under(&term(1), &labelled), lost);
    let labels: Vec<_> = three
        .iter()
        .map(|facts| store.by_key(&facts.key).map(|row| row.facts.labels.len()))
        .collect();
    assert_eq!(labels, [Some(1), Some(0), Some(0)]);

    // The worker comes back two wide. The term holds for the two rows
    // that stay and is lost ahead of the third rank's removal.
    assert_eq!(lay_under(&term(2), &ranked(2)), lost);
    let widths: Vec<_> = three
        .iter()
        .map(|facts| store.by_key(&facts.key).map(|row| row.facts.dp_size))
        .collect();
    assert_eq!(widths, [Some(2), Some(2), Some(3)]);
}

/// A row laid alone asks the term ahead of its write.
#[test]
fn a_row_laid_alone_under_a_lost_term_writes_nothing() {
    let (store, _) = door();
    let turn = TermFor(&store, std::cell::Cell::new(0));
    assert_eq!(
        lay_one(&turn, &Registrant::Control, Precondition::Any, URL, &|_| {
            stated()
        }),
        Err(MembershipWriteRefused::NotLeading(
            "the term moved".to_owned()
        ))
    );
    assert_eq!(store.all_rows(), (0, Vec::new()));
}

/// A lead lost behind an add's first row leaves that row. The same add
/// under a term that holds lays the rest, and the row that stands
/// answers unmoved.
#[test]
fn an_add_that_lost_its_lead_lays_the_rest_when_it_is_tried_again() {
    let (store, _) = door();
    let three = ranked(3);
    let add = |checks| {
        let turn = TermFor(&store, std::cell::Cell::new(checks));
        lay_as(&turn, Precondition::Absent, &three)
    };
    assert_eq!(
        add(1),
        Err(MembershipWriteRefused::NotLeading(
            "the term moved".to_owned()
        ))
    );
    assert_eq!(store.all_rows().1.len(), 1);
    let wrote = add(3);
    assert!(
        matches!(
            wrote.as_deref(),
            Ok([
                MembershipWriteOutcome::Unmoved(_),
                MembershipWriteOutcome::Registered(_),
                MembershipWriteOutcome::Registered(_)
            ])
        ),
        "{wrote:?}"
    );
    assert_eq!(store.all_rows().1.len(), 3);
}
