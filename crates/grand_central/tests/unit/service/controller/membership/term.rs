//! The cases of a call whose term ended ahead of its write: a drain, a
//! connect signal and a change in place write nothing, the replica's own
//! drain is refused as not leading, and the holder's two public entries
//! write under no term.

use super::*;

/// A drain goes with the word of the term its call was admitted under.
/// Here the fence names the term and the store takes the next one ahead
/// of the write, as a call finds it that waited for a lock through the
/// end of its term. The drain writes no status, on the object or in the
/// table, and the call answers that the replica does not lead, never
/// that the row moved. Under the term the log leads, the same drain is
/// written.
#[test]
fn a_drain_whose_call_lost_its_term_writes_nothing_and_is_refused_as_not_leading() {
    let (store, door) = door();
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    let id = WorkerId(row.id);
    let drain = proto::DeregisterWorkerRequest {
        worker_id: row.id,
        revision: row.revision,
        drain: true,
    };
    let lost = || {
        store::LeasesStore::set_term(store.as_ref(), TERM + 1);
        Ok(TERM)
    };
    let epoch = store.epoch();
    let refused = door
        .deregister(&operator(), lost, &drain)
        .expect_err("the term ended ahead of the write");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::Unavailable, super::super::LEADS_NO_TERM)
    );
    let status = |row: Option<WorkerRow>| row.map(|row| row.status);
    let pending = Some(WorkerStatus::Pending);
    assert_eq!(
        (status(store.get(id)), status(door.row(id))),
        (pending, pending)
    );
    assert_eq!(store.epoch(), epoch);
    door.deregister(&operator(), under(TERM + 1), &drain)
        .expect("draining");
    assert_eq!(status(store.get(id)), Some(WorkerStatus::Draining));
}

/// A connect signal goes with the word of its call's term too. This
/// registration loses its lead as its own write lands, so the signal
/// for its row, whose checks are off, writes nothing: the row stays
/// pending on the object and in the table, and the call that waits
/// answers it so. The registration sent again states equal facts and
/// writes no row. Under a term the log leads it sends the signal: the
/// row is ready under its id, at the revision it stood at.
#[test]
fn a_connect_signal_whose_call_lost_its_term_leaves_the_row_pending() {
    let waits = proto::RegisterWorkerRequest {
        wait_ready: true,
        ..ask(unchecked(URL), Mode::Add)
    };
    let (store, door, told) = lost_its_lead_at(0, &waits);
    let row = row_of(&told.expect("registered"));
    assert_eq!(row.status, proto::WorkerStatus::Pending as i32);
    let id = WorkerId(row.id);
    let status = |row: Option<WorkerRow>| row.map(|row| row.status);
    let pending = Some(WorkerStatus::Pending);
    assert_eq!(
        (status(store.get(id)), status(door.row(id))),
        (pending, pending)
    );
    assert_eq!(store.epoch(), 1, "the registration's write alone");
    // The registration sent again finds the row unmoved, and its signal
    // has the word of its own term. The log leads a later term now: a
    // call admitted under the one that ended sends nothing.
    store::LeasesStore::set_term(store.as_ref(), TERM + 2);
    let again = proto::RegisterWorkerRequest {
        wait_ready: true,
        ..ask(unchecked(URL), Mode::Upsert)
    };
    let ended = door
        .register(&operator(), under(TERM), &again)
        .expect("unmoved");
    assert_eq!(row_of(&ended), row);
    assert_eq!((status(store.get(id)), store.epoch()), (pending, 1));
    // One admitted under the term the log leads sends the signal, and
    // as it waits it answers the row behind it: ready, with no row
    // written and no life made.
    let told = door
        .register(&operator(), under(TERM + 2), &again)
        .expect("unmoved");
    assert!(matches!(told.event, Some(Event::Relabelled(_))), "{told:?}");
    let ready = row_of(&told);
    assert_eq!(
        (ready.id, ready.revision, ready.status),
        (row.id, row.revision, proto::WorkerStatus::Ready as i32)
    );
    let stored = store.get(id).expect("the row");
    assert_eq!(
        (stored.status, stored.revision),
        (WorkerStatus::Ready, row.revision)
    );
    assert_eq!(status(door.row(id)), Some(WorkerStatus::Ready));
    assert_eq!(
        (store.epoch(), store.all_rows().1.len()),
        (2, 1),
        "the status alone"
    );
    // Sent once more it finds the row ready and writes nothing.
    let once_more = door
        .register(&operator(), under(TERM + 2), &again)
        .expect("unmoved");
    assert_eq!((row_of(&once_more), store.epoch()), (ready, 2));
}

/// A registration that changes its row in place asks no term of its own
/// ahead of its write: the write's telling asks it. The log leads a later
/// term, and a call admitted under the one that ended would relabel the
/// waiting row. Its write is refused as not leading, and the row stands
/// as it stood: no label, pending, at its revision. The same registration
/// under the term the log leads changes the row in place and sends its
/// signal.
#[test]
fn a_change_in_place_whose_call_lost_its_term_writes_nothing() {
    let (store, door, _) = lost_its_lead_at(0, &ask(unchecked(URL), Mode::Add));
    store::LeasesStore::set_term(store.as_ref(), TERM + 2);
    let stood = store.by_key(URL).expect("the row");
    let relabelled = || {
        let labelled = proto::WorkerSpec {
            labels: [("tier".to_owned(), "gold".to_owned())].into(),
            ..unchecked(URL)
        };
        ask(labelled, Mode::Upsert)
    };
    let refused = door
        .register(&operator(), under(TERM), &relabelled())
        .expect_err("the term ended ahead of the write");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::Unavailable, super::super::LEADS_NO_TERM)
    );
    assert_eq!(store.get(stood.id).as_ref(), Some(&stood));
    door.register(&operator(), under(TERM + 2), &relabelled())
        .expect("in place");
    let ready = store.get(stood.id).expect("the row");
    assert_eq!(
        (ready.status, ready.revision, ready.facts.labels.len()),
        (WorkerStatus::Ready, stood.revision + 1, 1)
    );
}

/// The replica's own drain asks its turn's term, as a control call's
/// does. Where the term ended ahead of the write, the drain writes
/// nothing and is refused as a turn that does not lead.
#[test]
fn a_replicas_own_drain_whose_term_ended_is_refused_as_not_leading() {
    let (store, door) = door();
    let door = Arc::new(door);
    let maker = found("pods/a");
    ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM))
        .found_one(&maker, URL, &|_| found_facts())
        .expect("registered");
    let row = store.by_key(URL).expect("the registered row");
    let (seen, epoch) = (standing(&row), store.epoch());
    let taken = Arc::clone(&store);
    let hosted = ReplicaMembershipWriter::new(door, operator(), move || {
        store::LeasesStore::set_term(taken.as_ref(), TERM + 1);
        Ok(TERM)
    });
    assert_eq!(
        hosted.drained(Remover::Registrant(&maker), URL, seen),
        Err(MembershipWriteRefused::NotLeading(
            super::super::LEADS_NO_TERM.to_owned()
        ))
    );
    assert_eq!((store.by_key(URL), store.epoch()), (Some(row), epoch));
}

/// The holder's two public entries hand a word that always holds, as
/// before the door handed its own: a host with no term to end calls
/// them, and they write whatever a log leads. Over a log that left the
/// term, the door's two write nothing under it, and the public two
/// write.
#[test]
fn the_holders_public_connect_signal_and_drain_write_under_no_term() {
    let (store, door) = door();
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    let (id, revision) = (WorkerId(row.id), row.revision);
    store::LeasesStore::set_term(store.as_ref(), TERM + 1);
    let status = || store.get(id).map(|row| row.status);
    door.connect_row(id, revision, TERM);
    assert!(!door.drain_row(id, revision, TERM));
    assert_eq!(status(), Some(WorkerStatus::Pending));
    door.objects.connect_row(&store, id, revision);
    assert_eq!(status(), Some(WorkerStatus::Ready));
    assert!(door.objects.drain_row(&store, id, revision));
    assert_eq!(status(), Some(WorkerStatus::Draining));
}
