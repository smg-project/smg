//! A record made anew: the worker that left ahead of it, and the count of
//! leaves it begins past.

use super::*;

/// A worker that left behind a decision's read of the record is given no
/// place where the write meets a record made anew: the list's table was
/// read for a model whose profile reads no affinity, the record left the
/// map, the worker left while the map held no record of its model, and the
/// first model's next table made another. A record leaves at the timer's
/// sweep once it holds no place, and with its places where the follow of
/// a host's registry lets it go.
#[test]
fn a_worker_that_left_is_given_no_place_in_a_record_made_anew() {
    let mut document = PolicySpec::preset("least_load").expect("the preset");
    let reader = PolicySpec::preset("cache_aware").expect("the preset");
    let mut binding = reader.bindings[0].clone();
    binding.matcher.model = MODEL.into();
    document.bindings.insert(0, binding);
    document.profiles.extend(reader.profiles);
    for let_go in [false, true] {
        let spec = AppliedSpec::validate(&document).expect("the document validates");
        let (affinity, _, _) = hosted(false);
        let bed = Bed::new(affinity, spec, 4);
        let asked = prompt(1_000, 4);
        let facts = request(&asked, 7, NOW);
        let other = RequestFacts {
            model: "model-b",
            ..facts
        };
        if let_go {
            // The follow names a record that holds a place alone.
            let (column_of, first) = bed.column(&facts, Phase::Single);
            let first = first.expect("the record speaks for the list");
            let (list, id) = (&bed.pool, bed.ids[0]);
            record_by_column(list, Phase::Single, &facts, first, &column_of, id);
        }
        let (column_of, read) = bed.column(&facts, Phase::Single);
        let read = read.expect("the record speaks for the list");
        bed.column(&other, Phase::Single);
        if let_go {
            bed.affinity.let_go(&bed.affinity.unstated(|_| false));
        } else {
            bed.affinity.sweep();
        }
        assert!(bed.affinity.records.lock().is_empty(), "the record left");
        bed.affinity.forget(bed.ids[2]);
        bed.column(&facts, Phase::Single);
        let (list, id) = (&bed.pool, bed.ids[2]);
        record_by_column(list, Phase::Single, &facts, read, &column_of, id);
        let behind = held(&bed.column(&facts, Phase::Single).0);
        assert_eq!(behind, [(0, 0); 4], "let go: {let_go}");
    }
}

/// Two records that leave the map in one sweep with unequal counts of
/// leaves, four and one: the record made next begins past the greater,
/// whichever of the two the map holds first.
#[test]
fn a_record_made_anew_begins_past_the_greatest_count_that_left() {
    for (many, one) in [(MODEL, "model-b"), ("model-b", MODEL)] {
        let (affinity, _, _) = hosted(false);
        // Each record is the map's alone and holds no place.
        drop(affinity.record_of(many, 8, 1, false, None));
        for worker in 1..=3 {
            affinity.forget(WorkerId(worker));
        }
        drop(affinity.record_of(one, 8, 1, false, None));
        affinity.forget(WorkerId(4));
        affinity.sweep();
        assert!(affinity.records.lock().is_empty(), "the two left");
        let anew = affinity
            .record_of(one, 8, 1, false, None)
            .left
            .load(Ordering::Relaxed);
        assert_eq!(anew, 5, "{many} left with four");
    }
}
