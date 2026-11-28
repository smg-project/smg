//! The write of one row: a found worker registered under its maker, a
//! label written in place, and a changed fact that starts another life.
//! Equal facts over a checked row that is not draining write nothing. A
//! registrant that did not make a row is refused its registration, its
//! drain and its removal there.

use super::*;

#[test]
fn a_found_worker_is_registered_under_its_maker() {
    let (store, door) = door();
    let maker = Registrant::provider("pool", "first").expect("a maker");
    let wrote = door.found_one(&maker, URL, &|standing| {
        assert!(standing.is_none());
        stated()
    });
    let row = store.by_key(URL).expect("the registered row");
    assert_eq!(wrote, Ok(MembershipWriteOutcome::Registered(row.id)));
    assert_eq!(row.maker, maker);
    assert_eq!(
        row.facts,
        WorkerFacts {
            incarnation: membership::life(0, door.base, 0),
            ..stated()
        }
    );
    assert_eq!(
        (row.status, row.revision, store.epoch()),
        (WorkerStatus::Pending, 1, 1)
    );
}

#[test]
fn equal_facts_write_nothing() {
    let (store, door) = door();
    let maker = Registrant::provider("pool", "first").expect("a maker");
    door.found_one(&maker, URL, &|_| stated())
        .expect("registered");
    let row = store.by_key(URL).expect("the registered row");
    assert_eq!(
        door.found_one(&maker, URL, &|_| stated()),
        Ok(MembershipWriteOutcome::Unmoved(row.id))
    );
    let seen = standing(&row);
    assert_eq!(store.by_key(URL), Some(row));
    assert_eq!(store.epoch(), 1);

    assert!(matches!(
        door.gone(Remover::Registrant(&Registrant::Control), URL, seen),
        Err(MembershipWriteRefused::NotRegistrant(_))
    ));
    assert_eq!(store.epoch(), 1);
}

#[test]
fn a_changed_label_is_written_in_place_and_a_changed_role_is_another_life() {
    let (store, door) = door();
    let maker = Registrant::provider("pool", "first").expect("a maker");
    door.found_one(&maker, URL, &|_| stated())
        .expect("registered");
    let old = store.by_key(URL).expect("the registered row");
    let labelled = WorkerFacts {
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        ..stated()
    };
    assert_eq!(
        door.found_one(&maker, URL, &|_| labelled.clone()),
        Ok(MembershipWriteOutcome::InPlace(old.id))
    );
    let standing = store.by_key(URL).expect("the refreshed row");
    assert_eq!(standing.id, old.id);
    assert_eq!(standing.facts.incarnation, old.facts.incarnation);
    assert_eq!(standing.facts.labels, labelled.labels);
    assert_eq!(standing.revision, 2);

    let before = store.epoch();
    let wrote = door.found_one(&maker, URL, &|_| WorkerFacts {
        role: Role::Decode,
        ..labelled.clone()
    });
    let row = store.by_key(URL).expect("the replaced row");
    assert_eq!(
        wrote,
        Ok(MembershipWriteOutcome::Another {
            old: old.id,
            new: row.id
        })
    );
    assert_ne!(row.id, old.id);
    assert_eq!(store.get(old.id), None);
    assert_eq!(row.maker, maker);
    assert_eq!(
        row.facts.incarnation,
        membership::life(0, door.base, before)
    );
    assert_eq!((row.facts.role, row.revision), (Role::Decode, 1));
}

/// A refresh that changes the runtime a worker states starts another
/// life. The row that stated the old number is gone when the write is
/// answered. The new life is handed the old one's status: the worker's
/// checks are on, and none runs between the two.
#[test]
fn a_changed_demand_fact_is_another_life() {
    let (store, door) = door();
    let maker = Registrant::provider("pool", "first").expect("a maker");
    let stating = |runtime| WorkerFacts {
        demand_facts: DemandFacts {
            runtime,
            ..DemandFacts::default()
        },
        ..stated()
    };
    door.found_one(&maker, URL, &|_| stating(7))
        .expect("registered");
    // Its checks are on: a check's word makes the first life ready.
    let pending = store.by_key(URL).expect("the registered row");
    store
        .set_status(pending.id, WorkerStatus::Ready, pending.revision)
        .expect("ready");
    let old = store.by_key(URL).expect("the ready row");

    let wrote = door.found_one(&maker, URL, &|_| stating(9));
    let row = store.by_key(URL).expect("the replaced row");
    assert_eq!(
        wrote,
        Ok(MembershipWriteOutcome::Another {
            old: old.id,
            new: row.id
        })
    );
    assert_eq!(store.get(old.id), None);
    assert_eq!(
        row.facts.demand_facts,
        DemandFacts {
            runtime: 9,
            ..DemandFacts::default()
        }
    );
    assert_eq!(row.status, WorkerStatus::Ready, "handed on by the store");
}

#[test]
fn another_makers_row_refuses_every_write_and_stands() {
    let caller = Registrant::provider("pool", "second").expect("a maker");
    for (maker, why) in [
        (
            Registrant::Control,
            "the row was made by a control call: its maker alone changes it",
        ),
        (
            Registrant::provider("pool", "first").expect("a maker"),
            "the row is provider pool's under another key: that key's caller changes it",
        ),
    ] {
        let (store, door) = door();
        door.found_one(&maker, URL, &|_| stated())
            .expect("registered");
        let row = store.by_key(URL).expect("the registered row");
        let expected = Err(MembershipWriteRefused::NotRegistrant(why.to_owned()));
        assert_eq!(
            door.found_one(&caller, URL, &|_| panic!("only the maker states facts")),
            expected
        );
        assert_eq!(
            door.drained(Remover::Registrant(&caller), URL, standing(&row)),
            expected
        );
        assert_eq!(
            door.gone(Remover::Registrant(&caller), URL, standing(&row)),
            expected
        );
        assert_eq!(store.by_key(URL), Some(row));
        assert_eq!(store.epoch(), 1);
    }
}
