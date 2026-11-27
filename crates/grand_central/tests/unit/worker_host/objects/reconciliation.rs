//! A row against its object: the three acts of a write, the status copied
//! from the table, the entry under a row's two keys, and a cut laid whole.

use super::*;

/// One write of a replica's store, under the name a failure says.
type Write<'a> = (&'a str, &'a dyn Fn(&Replica));

/// The first act: a row written through the holder.
#[test]
fn a_row_that_arrives_has_its_object_built() {
    let mut replica = Replica::new();
    assert!(replica.objects.is_empty());
    let first = replica.write(facts(A, 1));
    let second = replica.write(facts(B, 1));
    let registry = Arc::clone(replica.objects.registry());
    assert_eq!((replica.objects.len(), registry.len()), (2, 2));
    assert_eq!(registry.get_by_model("m").len(), 2);
    for (url, row) in [(A, first), (B, second)] {
        let id = replica.id(url);
        assert_eq!(replica.at(url).url(), url);
        assert_eq!(replica.objects.object(row), Some(id.clone()));
        let revision = replica.row(url).revision;
        assert_eq!(
            replica.objects.standing(&id),
            Some(RowRevision { row, revision })
        );
    }
    let told = [("registered", replica.id(A)), ("registered", replica.id(B))];
    assert_eq!(replica.told(), told);
}

#[test]
fn the_row_is_answered_only_for_the_object_the_entry_was_written_for() {
    let replica = Replica::new();
    let row = replica.write(facts(A, 1));
    replica.write(facts(B, 1));
    let id = replica.id(A);
    let laid = replica.at(A);
    let standing = RowRevision {
        row,
        revision: replica.row(A).revision,
    };
    assert_eq!(replica.objects.standing_of(&id, &laid), Some(standing));
    assert_eq!(replica.objects.standing_of(&replica.id(B), &laid), None);
    assert_eq!(
        replica.objects.standing_of(&id, &object(&replica.row(A))),
        None
    );

    replica.patch(A, labels("gold"));
    assert_eq!(replica.objects.standing_of(&id, &laid), None);
    assert_eq!(
        replica.objects.standing_of(&id, &replica.at(A)),
        Some(RowRevision {
            row,
            revision: replica.row(A).revision,
        })
    );
}

/// The second act.
#[test]
fn a_row_that_goes_has_its_object_dropped() {
    let mut replica = Replica::new();
    let row = replica.write(facts(A, 1));
    replica.write(facts(B, 1));
    let id = replica.id(A);
    replica.told();
    replica.remove(A);
    let registry = replica.objects.registry();
    assert!(registry.get_by_url(A).is_none());
    assert_eq!((replica.objects.len(), registry.len()), (1, 1));
    assert!(!replica.objects.is_empty());
    assert_eq!(replica.objects.object(row), None);
    assert_eq!(replica.objects.standing(&id), None);
    assert_eq!(replica.told(), [("removed", id)]);
}

/// The third act, at each write of a row's facts: a patch, a refresh of
/// the registration, a new width. The id stands, the object is another,
/// and what ran on the old one runs on the new: never a removal and an
/// add.
#[test]
fn a_row_whose_facts_change_has_its_object_replaced_under_the_same_id() {
    let mut replica = Replica::new();
    let row = replica.write(facts(A, 1));
    let id = replica.id(A);
    replica.told();
    let wider = WorkerFacts {
        dp_size: 2,
        ..facts(A, 1)
    };
    let writes: [Write<'_>; 3] = [
        ("a patch", &|replica| replica.patch(A, labels("gold"))),
        ("a refresh", &|replica| {
            let (store, objects) = (replica.store.as_ref(), &replica.objects);
            let change = objects.upsert(store, facts(A, 1), Registrant::Control);
            assert_eq!(change, Some(MembershipChange::Relabelled { id: row }));
        }),
        ("a new width", &|replica| {
            let (store, objects) = (replica.store.as_ref(), &replica.objects);
            let change = objects.upsert(store, wider.clone(), Registrant::Control);
            assert_eq!(change, Some(MembershipChange::Resized { id: row }));
        }),
    ];
    let shown: [(Option<&str>, Option<usize>); 3] =
        [(Some("gold"), None), (None, None), (None, Some(2))];
    for (turn, ((name, write), shown)) in writes.into_iter().zip(shown).enumerate() {
        let before = replica.at(A);
        before.increment_load();
        write(&replica);
        let after = replica.at(A);
        assert!(!Arc::ptr_eq(&before, &after), "{name}: another object");
        assert_eq!(replica.id(A), id, "{name}: the same id");
        assert_eq!(after.load(), turn + 1, "{name}: the running state");
        assert_eq!(
            (replica.tier(A).as_deref(), after.dp_size()),
            shown,
            "{name}"
        );
        let revision = replica.row(A).revision;
        assert_eq!(
            replica.objects.standing(&id),
            Some(RowRevision { row, revision }),
            "{name}"
        );
        assert_eq!(replica.told(), [("replaced", id.clone())], "{name}");
    }
}

/// The holder's copy of the status a table holds, as a told status's
/// lay and a cut's call it.
#[test]
fn a_status_copied_from_the_table_makes_a_worker_available_only_while_ready() {
    let mut replica = Replica::new();
    replica.write(facts(A, 1));
    let row = replica.row(A);
    let copy = |replica: &Replica| replica.objects.copy_status(replica.store.as_ref(), row.id);
    let (id, worker) = (replica.id(A), replica.at(A));
    assert_eq!(worker.status(), ObjectStatus::Pending);
    assert!(!worker.is_available());
    assert_eq!(replica.told(), [("registered", id.clone())]);

    for (row_status, object_status, available) in [
        (WorkerStatus::Ready, ObjectStatus::Ready, true),
        (WorkerStatus::NotReady, ObjectStatus::NotReady, false),
    ] {
        assert!(replica
            .store
            .set_status(row.id, row_status, row.revision)
            .is_some());
        copy(&replica);
        assert!(Arc::ptr_eq(&worker, &replica.at(A)));
        assert_eq!(worker.status(), object_status);
        assert_eq!(worker.is_available(), available);
        // The registry's subscribers hear each change once.
        assert_eq!(replica.told(), [("status", id.clone())]);
    }

    for _ in 0..CircuitBreakerConfig::default().failure_threshold {
        worker.record_circuit_breaker_outcome(false);
    }
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
    assert!(replica
        .objects
        .registry()
        .set_worker_overloaded(&worker, true));
    assert!(worker.set_stall(Some(StallReason::Unreachable)));
    assert!(replica
        .store
        .set_status(row.id, WorkerStatus::Ready, row.revision)
        .is_some());
    replica.told();
    copy(&replica);
    assert_eq!(worker.status(), ObjectStatus::Ready);
    assert_eq!(replica.told(), [("status", id)]);
    // A promotion comes back as any promotion does: the open breaker
    // goes on trial. The host's own vetoes stand.
    assert_eq!(worker.circuit_breaker_state(), CircuitState::HalfOpen);
    assert!(worker.is_overloaded());
    assert_eq!(worker.stall_reason(), Some(StallReason::Unreachable));
    assert!(!worker.is_available());
}

#[test]
fn a_status_at_a_new_row_revision_waits_for_the_new_object() {
    let replica = Replica::new();
    let store = replica.store.as_ref();
    replica.write(facts(A, 1));
    let row = replica.row(A);
    let before = replica.at(A);
    let moved = store.set_status(row.id, WorkerStatus::Ready, row.revision);
    assert!(moved.is_some(), "the status moved");
    replica.patch_around(A, labels("gold"));

    replica.objects.copy_status(store, row.id);
    assert_eq!(before.status(), ObjectStatus::Pending);
    replica.lay_standing();
    let after = replica.at(A);
    assert!(!Arc::ptr_eq(&before, &after));
    assert_eq!(after.status(), ObjectStatus::Pending, "a lay copies none");
    replica.objects.copy_status(store, row.id);
    assert_eq!(after.status(), ObjectStatus::Ready);
}

#[test]
fn a_status_change_leaves_an_object_the_holder_did_not_lay_alone() {
    let replica = Replica::new();
    let store = replica.store.as_ref();
    replica.write(facts(A, 1));
    let row = replica.row(A);
    assert!(replica
        .store
        .set_status(row.id, WorkerStatus::Ready, row.revision)
        .is_some());
    replica.objects.copy_status(store, row.id);
    let registry = replica.objects.registry();
    let foreign = object(&row);
    assert!(registry.replace(&replica.id(A), Arc::clone(&foreign)));
    assert_eq!(foreign.status(), ObjectStatus::Ready);

    assert!(replica
        .store
        .set_status(row.id, WorkerStatus::NotReady, row.revision)
        .is_some());
    replica.objects.copy_status(store, row.id);
    assert_eq!(foreign.status(), ObjectStatus::Ready);
}

#[test]
fn a_cut_laid_whole_lays_the_status_of_both_kept_and_replaced_objects() {
    let mut replica = Replica::new();
    replica.write(facts(A, 1));
    replica.write(facts(B, 1));
    let (a, b) = (replica.at(A), replica.at(B));
    let (id_a, id_b) = (replica.id(A), replica.id(B));
    assert!(!a.is_available() && !b.is_available());
    replica.told();

    let other = MemoryStore::new();
    other.upsert(facts(A, 1));
    other.upsert(labelled(B, 1, "gold"));
    for url in [A, B] {
        let row = other.by_key(url).expect("a row in the snapshot");
        assert!(other
            .set_status(row.id, WorkerStatus::Ready, row.revision)
            .is_some());
    }
    let cut = other.checkpoint();
    assert!(replica.objects.restore_from(replica.store.as_ref(), cut));

    assert!(Arc::ptr_eq(&a, &replica.at(A)));
    assert!(a.is_available());
    assert!(!Arc::ptr_eq(&b, &replica.at(B)));
    assert!(replica.at(B).is_available());
    // Each object's change is told, kept or laid again.
    let told = replica.told();
    for id in [&id_a, &id_b] {
        assert!(told.contains(&("status", id.clone())), "{told:?}");
    }

    let row = other.by_key(A).expect("a row in the snapshot");
    assert!(other
        .set_status(row.id, WorkerStatus::NotReady, row.revision)
        .is_some());
    let cut = other.checkpoint();
    assert!(replica.objects.restore_from(replica.store.as_ref(), cut));
    assert!(Arc::ptr_eq(&a, &replica.at(A)));
    assert!(!a.is_available());
    assert_eq!(replica.told(), [("status", id_a)]);
}

#[test]
fn a_new_incarnation_of_a_key_drops_the_old_object_and_builds_one_under_a_new_id() {
    let mut replica = Replica::new();
    let old = replica.write(facts(A, 1));
    let (old_id, before) = (replica.id(A), replica.at(A));
    before.increment_load();
    replica.told();
    let new = replica.write(facts(A, 2));
    assert_ne!(old, new);
    let new_id = replica.id(A);
    assert_ne!(old_id, new_id);
    assert_eq!(replica.at(A).load(), 0, "nothing of the old object runs on");
    assert_eq!(replica.objects.len(), 1);
    assert_eq!(replica.objects.object(old), None);
    assert_eq!(replica.objects.standing(&old_id), None);
    assert_eq!(replica.objects.object(new), Some(new_id.clone()));
    let row = replica
        .objects
        .standing(&new_id)
        .map(|standing| standing.row);
    assert_eq!(row, Some(new));
    assert_eq!(
        replica.told(),
        [("removed", old_id), ("registered", new_id)]
    );
}

/// Models another writer of the registry set on a standing object read
/// back as set. A cut laid whole holds its row against the row read
/// back off the object. A row that states other models is laid again,
/// and the object in its place states the cut's row. The row beside it
/// keeps the very object.
///
/// Classic sets models the same way, in its model refresh
/// (`model_gateway/src/routers/common/worker_selection.rs:290`). It
/// builds no holder, so no cut is laid whole over its registry.
#[test]
fn a_cut_laid_whole_lays_a_row_again_where_models_were_set_on_its_object() {
    let mut replica = Replica::new();
    replica.write(facts(A, 1));
    replica.write(facts(B, 1));
    let (a, set, kept) = (replica.id(A), replica.at(A), replica.at(B));
    let card = openai_protocol::model_card::ModelCard::new;
    let registry = Arc::clone(replica.objects.registry());
    assert!(registry.refresh_models(&set, vec![card("m"), card("late")]));
    replica.told();

    let cut = replica.store.checkpoint();
    assert!(replica.objects.restore_from(&replica.store, cut));

    // A's object is another under its id, and states the row's models.
    assert_eq!(replica.id(A), a);
    assert!(!Arc::ptr_eq(&set, &replica.at(A)), "A is laid again");
    let row = replica.objects.row_of(&a).expect("A's row");
    assert_eq!(row.facts, replica.row(A).facts);
    // B's row reads back as stated: the very object.
    assert!(Arc::ptr_eq(&kept, &replica.at(B)), "B is kept");
    assert_eq!(replica.told(), [("replaced", a)]);
}

/// The entry keeps what stands for a row under two keys: the row's id and
/// its object's id. A reader by either key reads one remainder and one
/// object. Behind each write of the entry the pointers under the two keys
/// are the same allocations, for every row. The writes are a lay, a lay
/// in a standing object's place, a key's next life, a row stated again
/// and a row's end.
#[test]
fn a_rows_two_entries_name_one_remainder_and_one_object() {
    /// How many rows the entry holds, the same count each way, where each
    /// row's two entries name one remainder and one object.
    fn shared(replica: &Replica) -> usize {
        let held = replica.objects.held.read();
        for (row, (id, by_row)) in &held.objects {
            let laid = held.rows.get(id).expect("the object's entry");
            assert_eq!(laid.remainder.standing.row, *row);
            assert!(Arc::ptr_eq(&by_row.remainder, &laid.remainder), "{row:?}");
            assert!(Arc::ptr_eq(&by_row.object, &laid.object), "{row:?}");
        }
        assert_eq!(held.objects.len(), held.rows.len(), "one entry each way");
        held.rows.len()
    }

    let replica = Replica::new();
    for url in [A, B, C] {
        replica.write(facts(url, 1));
    }
    assert_eq!(shared(&replica), 3, "a lay of each row");
    replica.write(labelled(A, 1, "gold"));
    replica.write(facts(B, 2));
    assert_eq!(shared(&replica), 3, "a lay in place and a next life");
    // The same facts and revision, admitted at another time.
    let mut row = replica.row(C);
    row.admitted_at += Duration::from_secs(1);
    replica.objects.restate(&row);
    let stated = replica.objects.remainder(&replica.id(C));
    let stated = stated.expect("C's remainder");
    assert_eq!(stated.admitted_at, row.admitted_at, "stated again");
    assert_eq!(shared(&replica), 3, "a row stated again");
    replica.remove(A);
    assert_eq!(shared(&replica), 2, "a row's end");
}

/// A cut laid whole reads every row once: one write of each kind went
/// around the holder ahead of it. The objects of
/// the rows that went leave first, then each row that does not stand as
/// its object was built is laid again.
/// A row only a status write touched keeps the very object, and takes
/// the cut's status. A
/// registration refreshed with equal facts moved its revision and is
/// laid again. So is the row that left and came
/// back under its id and its revision with a new label: its facts moved.
#[test]
fn a_cut_laid_whole_drops_the_rows_that_went_and_lays_the_moved_rows_again() {
    let mut replica = Replica::new();
    let gone = replica.write(facts(A, 1));
    replica.write(facts(B, 1));
    let replaced = replica.write(facts(C, 1));
    replica.write(facts(E, 1));
    let back = replica.write(facts(F, 1));
    replica.write(facts(G, 1));
    let (a, b, c) = (replica.id(A), replica.id(B), replica.id(C));
    let (e, f, g) = (replica.id(E), replica.id(F), replica.id(G));
    let stood = replica.at(E);
    replica.told();

    replica.remove_around(A);
    replica.patch_around(B, labels("gold"));
    let next = replica.store.upsert(facts(C, 2)).id();
    let arrived = replica.store.upsert(facts(D, 1)).id();
    replica.remove_around(F);
    assert_eq!(replica.store.upsert(labelled(F, 1, "gold")).id(), back);
    assert_eq!(replica.row(F).revision, 1, "the first revision again");
    // A status write moves neither E's facts nor its revision.
    let unmoved = replica.row(E);
    assert!(replica
        .store
        .set_status(unmoved.id, WorkerStatus::NotReady, unmoved.revision)
        .is_some());
    assert_eq!(replica.row(E).revision, unmoved.revision);
    // A registration refreshed with equal facts moves G's revision alone.
    let refreshed = replica.row(G);
    replica.store.upsert(facts(G, 1));
    assert_eq!(replica.row(G).facts, refreshed.facts);
    assert_ne!(replica.row(G).revision, refreshed.revision);
    let cut = replica.store.checkpoint();
    assert!(replica.objects.restore_from(&replica.store, cut));

    let registry = Arc::clone(replica.objects.registry());
    assert_eq!((replica.objects.len(), registry.len()), (6, 6));
    assert!(registry.get_by_url(A).is_none());
    assert_eq!(replica.objects.object(gone), None);
    assert_eq!(replica.objects.object(replaced), None);
    assert_eq!(replica.id(B), b);
    assert_eq!(replica.tier(B).as_deref(), Some("gold"));
    assert_eq!((replica.id(E), replica.id(F)), (e.clone(), f.clone()));
    assert_eq!(replica.tier(F).as_deref(), Some("gold"));
    // E stands as its object was built: the very object, and its entry
    // names the row's revision still. A cut's status is laid on it.
    assert!(Arc::ptr_eq(&stood, &replica.at(E)));
    assert_eq!(stood.status(), ObjectStatus::NotReady);
    let standing = replica.objects.standing(&e).expect("E's row");
    assert_eq!(standing.revision, replica.row(E).revision);
    // G's revision moved: an object in its place, built at that revision.
    assert_eq!(replica.id(G), g);
    let standing = replica.objects.standing(&g).expect("G's row");
    assert_eq!(standing.revision, replica.row(G).revision);
    let (c_next, d) = (replica.id(C), replica.id(D));
    assert_ne!(c_next, c);
    assert_eq!(replica.objects.object(next), Some(c_next.clone()));
    assert_eq!(replica.objects.object(arrived), Some(d.clone()));
    // The cut's one status that moved, E's, is told behind the lays.
    let mut told = replica.told();
    assert_eq!(told.pop(), Some(("status", e)));
    let left: HashSet<_> = told.iter().take(2).cloned().collect();
    assert_eq!(left, HashSet::from([("removed", a), ("removed", c)]));
    let laid: HashSet<_> = told.iter().skip(2).cloned().collect();
    let each = [
        ("replaced", b),
        ("registered", c_next),
        ("registered", d),
        ("replaced", f),
        ("replaced", g),
    ];
    assert_eq!((told.len(), laid), (7, HashSet::from(each)));
}

/// A holder reads the row's told status under the object's mutation
/// lock, never ahead of it. The worker registry's test entry pauses the
/// holder where it comes to that lock
/// (`WorkerRegistry::with_another_write_before_the_lock`), and an
/// adapter's write lands there: it moves the object and tells the store.
/// The holder then reads the status the adapter told, and the object
/// keeps it. A read made ahead of the lock would lay the older status
/// over the adapter's.
#[test]
fn a_holder_reads_the_status_an_adapter_told_while_it_stood_at_the_objects_lock() {
    let registry = Arc::new(WorkerRegistry::<Members>::new());
    let objects = Arc::new(Objects::new(Arc::clone(&registry)));
    let store = Arc::new(MemoryStore::default());
    let row = objects.upsert(&store, facts(A, 1), Registrant::Control);
    let row = row.expect("the store takes the write").id();
    let id = objects.object(row).expect("the row's object");
    let worker = registry.get(&id).expect("a registered worker");
    let revision = store.get(row).expect("the worker's row").revision;
    assert!(store
        .set_status(row, WorkerStatus::NotReady, revision)
        .is_some());
    objects.copy_status(&store, row);
    assert_eq!(worker.status(), ObjectStatus::NotReady);

    let adapter = HealthAdapter::new(
        Arc::clone(&objects),
        Arc::clone(&store),
        Arc::new(SoleLead),
        SoleLead::TERM,
    );
    registry.with_another_write_before_the_lock(
        |_| objects.copy_status(&store, row),
        |_| {
            let answer = adapter.connected(&id, worker.revision());
            assert_eq!(answer, Some((ObjectStatus::NotReady, ObjectStatus::Ready)));
        },
    );
    assert_eq!(
        store.get(row).map(|row| row.status),
        Some(WorkerStatus::Ready)
    );
    assert_eq!(
        registry.get_by_url(A).expect("the worker").status(),
        ObjectStatus::Ready
    );
}

/// The holder is the registry's one writer.
#[test]
fn an_object_another_writer_left_under_a_rows_url_gives_way_to_the_rows() {
    let mut replica = Replica::new();
    let registry = Arc::clone(replica.objects.registry());
    let foreign: Arc<dyn Worker> = Arc::new(<BasicWorkerBuilder>::new(A).build());
    let id = registry.register(Arc::clone(&foreign)).expect("a free URL");
    let row = replica.write(facts(A, 1));
    assert!(!Arc::ptr_eq(&foreign, &replica.at(A)));
    assert_eq!(replica.id(A), id);
    assert_eq!(replica.objects.object(row), Some(id.clone()));
    let standing = replica.objects.standing(&id).map(|standing| standing.row);
    assert_eq!(standing, Some(row));
    assert_eq!(
        replica.told(),
        [("registered", id.clone()), ("replaced", id)]
    );
    assert_eq!(format!("{:?}", replica.objects), "Objects { held: 1, .. }");
}

/// Another store's cut can hold under one id and one revision a row of
/// other facts: the row reads as the one held, and its facts tell it
/// apart.
#[test]
fn a_store_that_took_another_historys_rows_has_its_moved_rows_laid_again() {
    let mut replica = Replica::new();
    let row = replica.write(facts(A, 1));
    replica.write(facts(B, 1));
    replica.write(facts(D, 1));
    let (a, b) = (replica.id(A), replica.id(B));
    let stood = replica.at(D);
    replica.told();

    let other = MemoryStore::new();
    other.upsert(WorkerFacts {
        labels: BTreeMap::from([("tier".to_string(), "gold".to_string())]),
        ..facts(A, 1)
    });
    other.upsert(facts(C, 1));
    other.upsert(facts(D, 1));
    let cut = other.checkpoint();
    let theirs = other.by_key(A).expect("a row");
    assert_eq!((theirs.id, theirs.revision), (row, replica.row(A).revision));
    assert!(replica.objects.restore_from(&replica.store, cut));

    assert_eq!(replica.id(A), a);
    assert_eq!(replica.tier(A).as_deref(), Some("gold"));
    assert!(replica.objects.registry().get_by_url(B).is_none());
    assert_eq!(replica.objects.len(), 3);
    // D states the same facts in the other history: the very object.
    assert!(Arc::ptr_eq(&stood, &replica.at(D)));
    let c = replica.id(C);
    assert_eq!(
        replica.told(),
        [("removed", b), ("replaced", a), ("registered", c)]
    );
}

thread_local! {
    /// The holder and the row a case watches.
    static WATCHED: RefCell<Option<(Arc<Objects<Seen>>, WorkerId)>> =
        const { RefCell::new(None) };
    /// What each publish read of them.
    static READ: RefCell<Vec<Option<u64>>> = const { RefCell::new(Vec::new()) };
}

/// A generation that reads, where the registry publishes it inside its
/// own write, the revision the holder says the watched row's object was
/// built from.
#[derive(Debug)]
struct Seen(Arc<[Arc<dyn Worker>]>);

impl Generation for Seen {
    fn of(members: Arc<[Arc<dyn Worker>]>) -> Self {
        WATCHED.with_borrow(|watched| {
            if let Some((objects, row)) = watched {
                let built = objects
                    .object(*row)
                    .and_then(|id| objects.standing(&id))
                    .map(|standing| standing.revision);
                READ.with_borrow_mut(|read| read.push(built));
            }
        });
        Self(members)
    }

    fn members(&self) -> &Arc<[Arc<dyn Worker>]> {
        &self.0
    }
}

/// An object's entry is written before the registry tells of the
/// object, at a build and at a replacement: what the registry publishes
/// inside its write already reads the row's revision of that write.
#[test]
fn an_objects_entry_stands_before_the_registry_tells_of_it() {
    let store = MemoryStore::new();
    let objects = Arc::new(Objects::<Seen>::new(Arc::new(WorkerRegistry::new())));
    let (row, built) = (crate::store::identity(A, 1), 1);
    WATCHED.set(Some((Arc::clone(&objects), row)));
    let written = objects.upsert(&store, facts(A, 1), Registrant::Control);
    assert_eq!(written, Some(MembershipChange::Registered { id: row }));
    let at_build = READ.take();
    assert!(!at_build.is_empty(), "a registration publishes");
    assert!(
        at_build.iter().all(|read| *read == Some(built)),
        "{at_build:?}"
    );

    assert!(objects.patch(&store, row, labels("gold"), built).is_some());
    let replaced = store.get(row).expect("a row").revision;
    assert_ne!(replaced, built);
    let at_replacement = READ.take();
    assert!(!at_replacement.is_empty(), "a replacement publishes");
    assert!(
        at_replacement.iter().all(|read| *read == Some(replaced)),
        "{at_replacement:?}"
    );
    WATCHED.set(None);
}
