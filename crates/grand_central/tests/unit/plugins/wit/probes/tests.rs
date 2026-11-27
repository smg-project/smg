use super::*;

#[test]
fn every_record_probe_matches_its_descriptor() {
    let drift = drift();
    assert!(drift.is_empty(), "{}", drift.join("\n"));
}

/// The probe of a record tied to two structs reads the record's order
/// from its `order` list, and a field of either struct that the list and
/// `host` both leave out follows the list instead of vanishing: the
/// descriptor's test then names the record.
#[test]
fn a_field_the_order_leaves_out_follows_it() {
    struct Row {
        a: u32,
        b: u64,
        c: bool,
    }
    struct Column {
        x: u32,
        y: u16,
    }
    let row = Row {
        a: 0,
        b: 0,
        c: false,
    };
    let column = Column { x: 0, y: 0 };
    let probe = probe!("toy", before [],
            Row { a, b, c } = &row,
            Column { x, y } = &column,
            order [b, x, a] host [c]);
    assert_eq!(
        probe.fields,
        [("b", "u64"), ("x", "u32"), ("a", "u32"), ("y", "u16")]
    );
}

#[test]
fn the_probes_cover_every_record_that_mirrors_a_struct() {
    let probed: Vec<&str> = all().into_iter().map(|p| p.record).collect();
    let records: Vec<&str> = super::super::types::TYPES
        .iter()
        .filter(|i| matches!(i, super::super::Item::Record { .. }))
        .map(|i| i.name())
        .collect();
    for record in records {
        // The host composes `dispatched` and the guest `score-answer`:
        // neither mirrors a struct of the crate.
        assert!(
            matches!(record, "dispatched" | "score-answer") || probed.contains(&record),
            "{record} has no probe"
        );
    }
}
