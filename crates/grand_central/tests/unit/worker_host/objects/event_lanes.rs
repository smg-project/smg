//! A row's lane in the KV tables a host handed the holder. A row that asks
//! for cache events is given one under its first model while that model's
//! index has room. A holder handed no tables makes none.

use super::*;

/// [`facts`] of a row whose signals ask for cache events.
fn asking(key: &str, incarnation: u64) -> WorkerFacts {
    WorkerFacts {
        signals: crate::store::Signals {
            events: true,
            ..crate::store::Signals::default()
        },
        ..facts(key, incarnation)
    }
}

/// Whether each of `rows` holds a lane at rank 0 of `tables`.
fn lanes(tables: &KvTables, rows: &[WorkerId]) -> Vec<bool> {
    rows.iter().map(|row| tables.holds_lane(*row, 0)).collect()
}

#[test]
fn a_holder_handed_tables_keeps_a_lane_for_each_row_that_asks_for_events() {
    let tables = Arc::new(KvTables::default());
    let objects = Objects::new(Arc::new(WorkerRegistry::<Members>::new()))
        .with_kv_tables(Arc::clone(&tables));
    let store = MemoryStore::new();
    let write = |facts: WorkerFacts| {
        let change = objects.upsert(&store, facts, Registrant::Control);
        change.expect("the store takes the write").id()
    };
    let (a, b, quiet) = (write(asking(A, 1)), write(asking(B, 1)), write(facts(C, 1)));
    assert_eq!(lanes(&tables, &[a, b, quiet]), [true, true, false]);
    // A row stated again in place: one that stops asking gives its lane
    // up, one that asks takes one. No object is laid anew for either.
    write(facts(A, 1));
    write(asking(C, 1));
    assert_eq!(lanes(&tables, &[a, b, quiet]), [false, true, true]);
    // A replaced row's lane leaves with it, blocks and all, and its
    // next life under the same URL starts with an empty lane, unheard.
    assert_eq!(stored_chain(&tables, b, 1, &[1]), BatchOutcome::Applied);
    assert_eq!(tables.indexed_blocks(b, 0), 1);
    let again = write(asking(B, 2));
    assert_eq!(lanes(&tables, &[b, again]), [false, true]);
    assert_eq!(tables.indexed_blocks(again, 0), 0);
    assert_eq!(tables.cursor(again, 0), None, "unheard");
    // A removed row's lane leaves with it.
    let left = store.by_key(C).expect("the row");
    let removed = objects.remove(&store, left.id, left.revision);
    assert!(removed.is_some());
    assert_eq!(lanes(&tables, &[quiet, again]), [false, true]);
    // A row that stays keeps what was heard for it through a later
    // write that changes nothing of it.
    assert_eq!(stored_chain(&tables, again, 1, &[1]), BatchOutcome::Applied);
    write(asking(D, 1));
    assert_eq!(tables.indexed_blocks(again, 0), 1);
}

#[test]
fn a_holder_handed_no_tables_makes_no_lane() {
    let objects = Objects::new(Arc::new(WorkerRegistry::<Members>::new()));
    let store = MemoryStore::new();
    objects.upsert(&store, asking(A, 1), Registrant::Control);
    assert!(objects.kv_tables().is_none());
    assert!(
        objects.registry().get_by_url(A).is_some(),
        "the object is laid all the same"
    );
}
