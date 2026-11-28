//! The credentials the holder's keeper holds: which object is built with
//! one, and what a registration keeps and puts back. What laying again
//! reaches, and what a removal, a refused write and a cut forget.

use super::*;

const KEY_A: &str = "key-a";
const KEY_B: &str = "key-b";
const KEY_C: &str = "key-c";

#[test]
fn a_row_laid_behind_a_kept_credential_builds_an_object_that_holds_it() {
    let replica = Replica::new();
    replica.register(facts(A, 1), Some(KEY_A));
    assert!(replica.holds_credential(A, Some(KEY_A)));

    // A row that names no engine dials its key: the keeper works the
    // address out as the holder does.
    let keyed = WorkerFacts {
        engine: String::new(),
        ..facts(B, 1)
    };
    replica.register(keyed, Some(KEY_B));
    // A row no registration here stated takes none.
    replica.write(facts(C, 1));
    assert!(replica.holds_credential(A, Some(KEY_A)));
    assert!(replica.holds_credential(B, Some(KEY_B)));
    assert!(replica.holds_credential(C, None));
}

#[test]
fn a_holder_handed_no_keeper_builds_objects_without_a_credential() {
    let registry = Arc::new(WorkerRegistry::<Members>::new());
    let objects = Objects::new(Arc::clone(&registry));
    let store = MemoryStore::new();
    objects.upsert(&store, facts(A, 1), Registrant::Control);
    let laid = registry.get_by_url(A).expect("the row's object");
    assert!(laid.api_key().is_none());
    assert_eq!(objects.lay_again(&store, &facts(A, 1)), 0);
}

#[test]
fn each_rank_row_of_an_engine_holds_the_engines_credential() {
    let replica = Replica::new();
    replica.register(rank_facts(A, C, 0), Some(KEY_C));
    replica.register(rank_facts(B, C, 1), Some(KEY_C));
    assert!(replica.holds_credential(A, Some(KEY_C)));
    assert!(replica.holds_credential(B, Some(KEY_C)));
}

#[test]
fn a_rank_row_another_host_registered_takes_no_credential_until_it_is_stated_here() {
    let replica = Replica::new();
    replica.register(rank_facts(A, C, 0), Some(KEY_C));
    // The engine's other rank came through another host: no keep here.
    let other = replica.write(rank_facts(B, C, 1));
    assert!(replica.holds_credential(A, Some(KEY_C)));
    assert!(replica.holds_credential(B, None));

    // It is stated here too. A door writes nothing over equal facts,
    // and the acceptance says the address is to be laid again.
    let stated = replica.kept(&rank_facts(B, C, 1), Some(KEY_C));
    assert!(stated.accepted(other));
    assert_eq!(replica.lay_again(&rank_facts(B, C, 1)), 1);
    assert!(replica.holds_credential(B, Some(KEY_C)));
}

#[test]
fn a_registration_refreshed_behind_a_new_credential_is_laid_with_it() {
    let mut replica = Replica::new();
    let registered = facts(A, 1);
    replica.register(registered.clone(), Some(KEY_A));
    assert!(replica.holds_credential(A, Some(KEY_A)));
    let id = replica.id(A);
    replica.told();

    // The row is written again behind another credential, and the
    // holder's write builds its object while the registration is in
    // flight.
    let refreshed = replica.kept(&registered, Some(KEY_B));
    let row = replica.write(registered.clone());
    assert_eq!(replica.told(), [("replaced", id)]);
    assert!(replica.holds_credential(A, Some(KEY_B)));

    // The object already holds what its row takes.
    assert!(refreshed.accepted(row));
    assert_eq!(replica.lay_again(&registered), 0);
}

#[test]
fn a_removal_forgets_nothing_while_another_row_dials_the_address() {
    let replica = Replica::new();
    replica.register(rank_facts(A, C, 0), Some(KEY_C));
    replica.register(rank_facts(B, C, 1), Some(KEY_C));

    replica.remove(A);
    assert_eq!(replica.credentials.len(), 1);
    // The other rank's row is written again: it is bound still.
    replica.write(rank_facts(B, C, 1));
    assert!(replica.holds_credential(B, Some(KEY_C)));
    replica.remove(B);
    assert!(replica.credentials.is_empty());
}

#[test]
fn a_removal_keeps_what_a_registration_in_flight_stated() {
    let replica = Replica::new();
    replica.register(facts(A, 1), Some(KEY_A));
    assert!(replica.holds_credential(A, Some(KEY_A)));

    let next = replica.kept(&facts(A, 2), Some(KEY_B));
    replica.remove(A);
    // The removed row is forgotten. What the flight stated stays.
    assert_eq!(replica.credentials.len(), 1);
    let row = replica.write(facts(A, 2));
    if next.accepted(row) {
        replica.lay_again(&facts(A, 2));
    }
    assert!(replica.holds_credential(A, Some(KEY_B)));
}

#[test]
fn a_removal_after_a_registration_that_moved_nothing_forgets_the_credential() {
    // On either answer of the registration that moved nothing.
    for put_back in [true, false] {
        let replica = Replica::new();
        let row = replica.register(facts(A, 1), Some(KEY_A));

        // The same registration again: a door writes nothing over
        // equal facts, and nothing is to be laid again.
        let again = replica.kept(&facts(A, 1), Some(KEY_A));
        if put_back {
            assert!(!again.put_back());
        } else {
            assert!(!again.accepted(row));
        }

        replica.remove(A);
        assert!(replica.credentials.is_empty());
    }
}

#[test]
fn a_registration_refused_at_an_address_that_stands_leaves_the_standing_workers_credential() {
    let replica = Replica::new();
    replica.register(facts(A, 1), Some(KEY_A));

    let before = replica.at(A);
    assert!(!replica.kept(&facts(A, 1), Some(KEY_B)).put_back());
    assert!(Arc::ptr_eq(&before, &replica.at(A)));

    replica.write(facts(A, 1));
    assert!(!Arc::ptr_eq(&before, &replica.at(A)));
    assert!(replica.holds_credential(A, Some(KEY_A)));
}

#[test]
fn a_registration_that_moved_no_row_behind_a_new_credential_is_laid_again_with_it() {
    let mut replica = Replica::new();
    let row = replica.register(facts(A, 1), Some(KEY_A));
    let id = replica.id(A);
    let standing = replica.objects.standing(&id);
    replica.told();

    // The same worker registered with another credential: a door
    // writes nothing over equal facts, so no row moves and no object
    // is built. The acceptance says the address is to be laid again.
    let again = replica.kept(&facts(A, 1), Some(KEY_B));
    assert!(again.accepted(row));
    assert!(replica.holds_credential(A, Some(KEY_A)));

    // The holder lays the object again: one replacement under its id,
    // at the revision its row stands at.
    assert_eq!(replica.lay_again(&facts(A, 1)), 1);
    assert!(replica.holds_credential(A, Some(KEY_B)));
    assert_eq!(replica.told(), [("replaced", id.clone())]);
    assert_eq!(replica.objects.standing(&id), standing);

    // An object that holds what its row takes is not laid again.
    assert_eq!(replica.lay_again(&facts(A, 1)), 0);
    assert!(replica.told().is_empty());

    // A registration that states no credential changes it too, and
    // the object is laid again without one.
    assert!(replica.kept(&facts(A, 1), None).accepted(row));
    assert_eq!(replica.lay_again(&facts(A, 1)), 1);
    assert!(replica.holds_credential(A, None));
}

#[test]
fn an_object_built_with_a_refused_credential_is_laid_again_with_what_stood() {
    let replica = Replica::new();
    replica.register(facts(A, 1), Some(KEY_A));

    // The standing row is written again while a registration is in
    // flight: its object is built with the stated credential.
    let refused = replica.kept(&facts(A, 1), Some(KEY_B));
    replica.write(facts(A, 1));
    assert!(replica.holds_credential(A, Some(KEY_B)));

    // The put-back says so, and the holder lays the object again.
    assert!(refused.put_back());
    assert_eq!(replica.lay_again(&facts(A, 1)), 1);
    assert!(replica.holds_credential(A, Some(KEY_A)));
}

#[test]
fn a_row_built_while_a_first_registration_is_in_flight_loses_its_credential_at_a_put_back() {
    let replica = Replica::new();
    // The row came through another host while a registration here
    // stated a credential for its address and was then refused.
    let refused = replica.kept(&facts(A, 1), Some(KEY_A));
    replica.write(facts(A, 1));
    assert!(replica.holds_credential(A, Some(KEY_A)));

    assert!(refused.put_back());
    assert_eq!(replica.lay_again(&facts(A, 1)), 1);
    assert!(replica.holds_credential(A, None));
    assert!(replica.credentials.is_empty());
}

#[test]
fn a_registration_refused_behind_a_rotation_not_yet_laid_leaves_the_rotated_credential() {
    let replica = Replica::new();
    let row = replica.register(facts(A, 1), Some(KEY_A));

    // The credential alone changes. It is accepted, and the holder
    // has not laid the address again yet.
    assert!(replica.kept(&facts(A, 1), Some(KEY_B)).accepted(row));
    // The next registration states the first credential again. The
    // holder lays the address meanwhile: the object already holds
    // what is stated, and the read marks the flight.
    let refused = replica.kept(&facts(A, 1), Some(KEY_A));
    assert_eq!(replica.lay_again(&facts(A, 1)), 0);
    // Refused: the rotated credential stands again, and the put-back
    // says an object holds the refused one.
    if refused.put_back() {
        replica.lay_again(&facts(A, 1));
    }
    assert!(replica.holds_credential(A, Some(KEY_B)));
}

/// One registration that keeps its credential at a time: its write
/// runs inside the holder's keeping turn, and the turn is let go
/// behind it.
#[test]
fn a_keeping_write_runs_in_a_turn_of_its_own() {
    let replica = Replica::new();
    let objects = &replica.objects;
    let held = objects.keeping(A, None, || objects.keeping.is_locked());
    assert_eq!(held, Ok(true), "the write runs inside the turn");
    assert!(!objects.keeping.is_locked(), "let go behind the write");
}

/// A registration that keeps, whose door answers and names no row the
/// registration stands on, is put back. The keep stands ahead of the
/// write and inside the turn. Nothing is kept behind it, the door's
/// answer comes back whole, and the turn is let go.
#[test]
fn a_keeping_registration_whose_answer_names_no_row_is_put_back() {
    let replica = Replica::new();
    let (objects, credentials) = (&replica.objects, &replica.credentials);
    let stated = facts(A, 1);
    let wrote = objects.register_keeping(
        replica.store.as_ref(),
        Some((&stated, Some(KEY_A.to_owned()))),
        (A, None),
        || {
            assert!(
                objects.keeping.is_locked(),
                "the write runs inside the turn"
            );
            assert_eq!(credentials.len(), 1, "kept ahead of the write");
            Ok(Vec::new())
        },
    );
    assert_eq!(wrote, Ok(Ok(Vec::new())));
    assert!(credentials.is_empty(), "put back");
    assert!(!objects.keeping.is_locked(), "let go behind the answer");
}

#[test]
fn laying_again_reaches_each_rank_row_that_dials_the_address_and_no_other() {
    let replica = Replica::new();
    let first = replica.register(rank_facts(A, C, 0), Some(KEY_C));
    replica.register(rank_facts(B, C, 1), Some(KEY_C));
    replica.register(facts(D, 1), Some(KEY_A));
    let other = replica.at(D);

    // One rank's registration states another credential: it is the
    // engine's, for each rank row bound there.
    let rotated = replica.kept(&rank_facts(A, C, 0), Some(KEY_B));
    assert!(rotated.accepted(first));
    assert_eq!(replica.lay_again(&rank_facts(A, C, 0)), 2);
    assert!(replica.holds_credential(A, Some(KEY_B)));
    assert!(replica.holds_credential(B, Some(KEY_B)));
    assert!(Arc::ptr_eq(&other, &replica.at(D)));

    // An address no row dials lays nothing.
    assert_eq!(replica.lay_again(&facts(E, 1)), 0);
}

/// Laying again reads no store. A row a registration wrote around the
/// holder has no object for the call to lay, and the keeper forgets
/// it: its object, laid where a cut's rows are laid, takes no
/// credential. A row
/// written through the holder is laid again from the row its object
/// states, not from the store's.
#[test]
fn laying_again_reads_the_objects_and_no_store() {
    let replica = Replica::new();
    let token = replica.kept(&facts(A, 1), Some(KEY_A));
    let row = replica.store.upsert(facts(A, 1)).id();
    assert!(token.accepted(row));
    assert_eq!(replica.lay_again(&facts(A, 1)), 0);
    assert!(replica.objects.is_empty(), "the call laid nothing");
    assert!(replica.credentials.is_empty(), "a row with no object");
    replica.lay_standing();
    assert!(replica.holds_credential(A, None));

    // Through the holder, then a rotation, then the store's row moved
    // around the holder: the call lays the row the object states.
    let row = replica.register(facts(B, 1), Some(KEY_A));
    assert!(replica.kept(&facts(B, 1), Some(KEY_B)).accepted(row));
    let late = WorkerPatch {
        priority: Some(7),
        ..WorkerPatch::default()
    };
    replica.patch_around(B, late);
    assert_eq!(replica.lay_again(&facts(B, 1)), 1);
    assert!(replica.holds_credential(B, Some(KEY_B)));
    let standing = replica.objects.standing(&replica.id(B));
    assert_eq!(standing.map(|standing| standing.revision), Some(1));
}

/// A removal that lands ahead of its registration's answer finds no
/// bound row to forget. The answer then binds a row that left. It
/// says the address is to be laid again, and that call finds no object
/// for the row and has the keeper forget it.
#[test]
fn laying_again_forgets_a_row_bound_after_it_left() {
    let replica = Replica::new();
    let store = replica.store.as_ref();
    let token = replica.kept(&facts(A, 1), Some(KEY_A));
    let row = replica.write(facts(A, 1));
    let removed = replica.objects.remove(store, row, 1);
    assert!(removed.is_some(), "the holder's own removal");
    assert!(token.accepted(row), "a row not bound before");
    assert_eq!(replica.credentials.len(), 1, "bound behind its leaving");
    assert_eq!(replica.lay_again(&facts(A, 1)), 0);
    assert!(replica.credentials.is_empty());
}

/// A write through the holder that takes a row out has the keeper
/// forget that row in the same call: a removal its
/// row, and a key's next life the old life's row.
#[test]
fn a_write_that_takes_a_row_out_forgets_it() {
    let replica = Replica::new();
    let store = replica.store.as_ref();
    let row = replica.register(facts(A, 1), Some(KEY_A));
    replica.register(facts(B, 1), Some(KEY_B));
    assert_eq!(replica.credentials.len(), 2);
    assert!(replica.objects.remove(store, row, 1).is_some());
    assert_eq!(replica.credentials.len(), 1, "the removed row's alone");
    // The other key's next life, stated with no keep.
    let next = replica
        .objects
        .upsert(store, facts(B, 2), Registrant::Control);
    assert!(matches!(next, Some(MembershipChange::Replaced { .. })));
    assert!(replica.credentials.is_empty(), "the old life's row");
    assert!(replica.holds_credential(B, None));
}

#[test]
fn a_registration_with_no_credential_clears_one_a_removed_worker_left() {
    let replica = Replica::new();
    replica.register(facts(A, 1), Some(KEY_A));
    replica.remove(A);

    replica.register(facts(A, 1), None);
    assert!(replica.holds_credential(A, None));
}

/// A cut laid whole forgets the rows that left, and a refused write
/// forgets none. A registration decided over a row that left around
/// the holder is refused, and the keeper is asked about no row. So a
/// row a registration bound and the holder has not read yet keeps its
/// credential, and its object is built with it. A cut that drops a row
/// ends its object, and the keeper drops every row that left. A
/// credential goes with its address's last row.
#[test]
fn a_cut_laid_whole_forgets_the_rows_that_left_and_a_refused_write_none() {
    let replica = Replica::new();
    let store = replica.store.as_ref();
    replica.register(facts(A, 1), Some(KEY_A));
    // A second registration, bound and not read by the holder yet.
    let token = replica.kept(&facts(B, 1), Some(KEY_B));
    assert!(token.accepted(store.upsert(facts(B, 1)).id()));
    // Another writer's removal, then the holder's own write of the key.
    replica.remove_around(A);
    let labelled = WorkerFacts {
        labels: BTreeMap::from([("tier".to_owned(), "gold".to_owned())]),
        ..facts(A, 1)
    };
    let refused = replica.objects.upsert(store, labelled, Registrant::Control);
    assert_eq!(refused, None, "decided over a row that left");
    assert_eq!(
        replica.credentials.len(),
        2,
        "a refused write forgets nothing"
    );
    replica.lay_standing();
    assert!(replica.holds_credential(B, Some(KEY_B)), "the unread row's");

    // A cut with no row.
    let cut = MemoryStore::new().checkpoint();
    assert!(replica.objects.restore_from(store, cut));
    assert!(replica.objects.is_empty(), "the cut dropped the row");
    assert!(replica.credentials.is_empty(), "behind the cut's read");
}

/// A refused write forgets nothing, whatever it decided: a change in
/// place, a new width, a key's next life, a removal. The row it was
/// decided over left around the holder. Its object stands as it stood
/// with its credential, and a removal's is laid again with it. A cut
/// laid whole ends the object, and the keeper forgets the row.
#[test]
fn a_refused_write_of_each_kind_forgets_nothing_and_a_cut_forgets_the_row() {
    type Refused = fn(&Replica, WorkerId) -> Option<MembershipChange>;
    fn lay(replica: &Replica, facts: WorkerFacts) -> Option<MembershipChange> {
        let store = replica.store.as_ref();
        replica.objects.upsert(store, facts, Registrant::Control)
    }
    let writes: [(&str, Refused); 4] = [
        ("a change in place", |replica, _| {
            lay(replica, labelled(A, 1, "gold"))
        }),
        ("a new width", |replica, _| {
            let wider = WorkerFacts {
                dp_size: 2,
                ..facts(A, 1)
            };
            lay(replica, wider)
        }),
        ("a key's next life", |replica, _| lay(replica, facts(A, 2))),
        ("a removal", |replica, row| {
            replica.objects.remove(replica.store.as_ref(), row, 1)
        }),
    ];
    for (name, write) in writes {
        let replica = Replica::new();
        let row = replica.register(facts(A, 1), Some(KEY_A));
        // Another writer's removal, around the holder.
        replica.remove_around(A);
        assert_eq!(write(&replica, row), None, "{name}: over a row that left");
        assert_eq!(replica.credentials.len(), 1, "{name}: nothing forgotten");
        assert!(replica.holds_credential(A, Some(KEY_A)), "{name}");
        let store = replica.store.as_ref();
        assert!(replica.objects.restore_from(store, store.checkpoint()));
        assert!(replica.credentials.is_empty(), "{name}: behind the cut");
    }
}
