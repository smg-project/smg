use super::*;

/// The type's list of every kind, held to the order of the bits. The
/// match names each, so a new kind does not compile until it has its
/// place here, which is its place in [`EventKind::ALL`]: the one list
/// that the set of every kind, the set's print and the cases read.
fn kinds() -> [EventKind; 6] {
    let kinds = EventKind::EVERY;
    for (at, kind) in kinds.into_iter().enumerate() {
        let place = match kind {
            EventKind::Issued => 0,
            EventKind::Dispatched => 1,
            EventKind::Completed => 2,
            EventKind::Released => 3,
            EventKind::Membership => 4,
            EventKind::Signal => 5,
        };
        // Its place in the list is its bit.
        assert_eq!((place, EventSet::of([kind]).0), (at, 1 << at), "{kind:?}");
    }
    kinds
}

/// A load word made by its call states no named value. It takes a
/// report's values whole and keeps every other field. Two words of one
/// row are two where their values are. A word of another kind takes no
/// value.
#[test]
fn a_load_word_is_made_with_no_value_and_takes_its_reports_values() {
    use crate::load::SignalCell;
    let row = LoadRow {
        waiting_uncached_tokens: 7,
        ..LoadRow::default()
    };
    let load = |signals| Signal::Load {
        worker: WorkerId(3),
        dp_rank: 1,
        row,
        sampled_at_micros: 9,
        signals,
    };
    let made = Signal::load(WorkerId(3), 1, row, 9);
    assert_eq!(made, load(ReportSignals::NONE));
    let cells = [
        SignalCell {
            value: 7.0,
            sampled_at_micros: 5,
        },
        SignalCell::NONE,
    ];
    let stated = ReportSignals::new(&cells);
    assert_eq!(made.with_signals(stated), load(stated));
    assert_ne!(made.with_signals(stated), made, "one row, two words");
    let health = Signal::Health {
        worker: WorkerId(3),
        health: Health::default(),
    };
    assert_eq!(health.with_signals(stated), health);
}

#[test]
fn an_event_set_holds_the_kinds_it_was_made_of_and_no_other() {
    let kinds = kinds();
    assert_eq!(EventSet::default(), EventSet::NONE);
    assert!(EventSet::NONE.is_empty() && !EventSet::ALL.is_empty());
    assert_eq!(EventSet::of(kinds), EventSet::ALL);
    for kind in kinds {
        let alone = EventSet::of([kind]);
        assert!(EventSet::ALL.contains(kind) && !EventSet::NONE.contains(kind));
        for other in kinds {
            assert_eq!(alone.contains(other), other == kind, "{kind:?} {other:?}");
        }
    }
}

/// A set prints the kinds it holds by name, in the order of the bits: a
/// row in a log line or a failing case names what the row lists.
#[test]
fn an_event_set_prints_the_kinds_it_holds_by_name() {
    let two = EventSet::of([EventKind::Signal, EventKind::Completed]);
    assert_eq!(format!("{two:?}"), "{Completed, Signal}");
    assert_eq!(format!("{:?}", EventSet::NONE), "{}");
    assert_eq!(
        format!("{:?}", EventSet::ALL),
        "{Issued, Dispatched, Completed, Released, Membership, Signal}"
    );
}

#[test]
fn a_pair_books_one_lease_per_side_on_its_own_target() {
    let prefill = Lease {
        side: Side::Prefill,
        target_index: 0,
        kind: LeaseKind::Reservation,
        ..Lease::default()
    };
    let decode = Lease {
        side: Side::Decode,
        target_index: 0,
        ..prefill
    };
    assert_ne!(prefill.side, decode.side);
    assert_eq!(prefill.target_index, decode.target_index);
    assert_eq!(decode.kind, LeaseKind::Reservation);
}

/// An outcome made by its constructor states the status and the class
/// it was given and nothing else: every measure is zero, and it names
/// no side and no served worker. It is the literal it stands for.
#[test]
fn an_outcome_reported_states_its_end_and_no_measure() {
    let reported = Outcome::reported(503, TransportClass::ServerError);
    let literal = Outcome {
        side: Side::Single,
        served: WorkerId(0),
        status: 503,
        class: TransportClass::ServerError,
        ttft_micros: 0,
        tokens_in: 0,
        tokens_out: 0,
        duration_micros: 0,
    };
    assert_eq!(reported, literal);
}
