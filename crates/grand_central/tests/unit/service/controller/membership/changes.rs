//! The cases of a patch, a drain and a removal over a row that stands: the
//! revision a patch merges under, what a drain and a removal answer, the
//! registration a drain ends with, and what a row's revision counts.

use super::*;

#[test]
fn a_patch_merges_under_the_revision_its_caller_read() {
    let (store, door) = door();
    let labelled = proto::WorkerSpec {
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        ..spec(URL)
    };
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &ask(labelled, Mode::Add))
            .expect("registered"),
    );
    let patch = proto::PatchWorkerRequest {
        worker_id: row.id,
        labels: [("lane".to_owned(), "b".to_owned())].into(),
        remove_labels: vec!["tier".to_owned()],
        overload: None,
        revision: row.revision,
    };
    let told = door
        .patch(&operator(), under(TERM), &patch)
        .expect("patched");
    assert!(matches!(told.event, Some(Event::Relabelled(_))));
    let now = row_of(&told);
    assert_eq!(now.labels, [("lane".to_owned(), "b".to_owned())].into());
    assert_eq!(
        (now.revision, told.membership_epoch, told.seq, told.term),
        (2, 2, 3, TERM)
    );

    // The same patch under the revision it read: the row moved.
    let stale = door
        .patch(&operator(), under(TERM), &patch)
        .expect_err("a stale revision");
    assert_eq!(stale.code(), Code::FailedPrecondition);
    assert_eq!(stale.message(), "the row is at revision 2");

    // Under the row's revision it changes nothing and writes nothing.
    let again = proto::PatchWorkerRequest {
        revision: 2,
        ..patch.clone()
    };
    let told = door
        .patch(&operator(), under(TERM), &again)
        .expect("answered");
    assert_eq!(
        (row_of(&told).revision, told.membership_epoch, told.seq),
        (2, 2, 3)
    );
    assert_eq!(store.epoch(), 2);

    let nobody = proto::PatchWorkerRequest {
        worker_id: 1,
        ..patch
    };
    let absent = door
        .patch(&operator(), under(TERM), &nobody)
        .expect_err("no such worker");
    assert_eq!(absent.code(), Code::NotFound);
}

#[test]
fn a_drain_answers_the_status_change_and_a_removal_the_row_as_it_last_stood() {
    let (store, door) = door();
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    let stale = proto::DeregisterWorkerRequest {
        worker_id: row.id,
        revision: row.revision + 1,
        drain: false,
    };
    let refused = door
        .deregister(&operator(), under(TERM), &stale)
        .expect_err("a stale revision");
    assert_eq!(refused.code(), Code::FailedPrecondition);

    let drain = proto::DeregisterWorkerRequest {
        worker_id: row.id,
        revision: row.revision,
        drain: true,
    };
    let told = door
        .deregister(&operator(), under(TERM), &drain)
        .expect("draining");
    assert_eq!(
        told.event,
        Some(Event::Status(proto::StatusChanged {
            worker_id: row.id,
            old: proto::WorkerStatus::Pending as i32,
            new: proto::WorkerStatus::Draining as i32,
            health: proto::HealthFlag::Draining as u32,
        }))
    );
    assert_eq!((told.membership_epoch, told.seq), (2, 3));
    // A second drain writes nothing and answers the row as it stands,
    // under the revision its registration gave: no status change moved
    // it.
    let told = door
        .deregister(&operator(), under(TERM), &drain)
        .expect("draining still");
    let stood = store.get(WorkerId(row.id)).expect("the row stands");
    assert_eq!(
        told.event,
        Some(Event::Relabelled(super::super::worker_wire::row(&stood)))
    );
    assert_eq!(
        (
            stood.status,
            stood.revision,
            told.membership_epoch,
            told.seq
        ),
        (WorkerStatus::Draining, row.revision, 2, 3)
    );

    let removal = proto::DeregisterWorkerRequest {
        drain: false,
        ..drain
    };
    let told = door
        .deregister(&operator(), under(TERM), &removal)
        .expect("removed");
    assert_eq!(
        told.event,
        Some(Event::Removed(proto::Removed {
            worker_id: row.id,
            registration_key: URL.to_owned(),
            revision: 1,
            draining: true,
        }))
    );
    assert_eq!((told.membership_epoch, told.seq, store.epoch()), (3, 4, 3));
    assert_eq!(store.by_key(URL), None);
    let gone = door
        .deregister(&operator(), under(TERM), &removal)
        .expect_err("gone");
    assert_eq!(gone.code(), Code::NotFound);
}

#[test]
fn a_drain_ends_with_the_registration_it_named() {
    let (store, door) = door();
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Add))
            .expect("registered"),
    );
    let drain = proto::DeregisterWorkerRequest {
        worker_id: row.id,
        revision: row.revision,
        drain: true,
    };
    door.deregister(&operator(), under(TERM), &drain)
        .expect("draining");
    // A patch is the same registration: draining still.
    let patch = proto::PatchWorkerRequest {
        worker_id: row.id,
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        remove_labels: Vec::new(),
        overload: None,
        revision: row.revision,
    };
    let told = door
        .relabel(&operator(), &Registrant::Control, TERM, &patch)
        .expect("patched");
    assert_eq!(row_of(&told).status, proto::WorkerStatus::Draining as i32);
    // A registration over a draining row is the URL's next incarnation,
    // by an upsert as by a replacement: the drain is not handed on, and
    // an unchecked worker is ready from its own.
    let told = door
        .register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Upsert))
        .expect("registered anew");
    let Some(Event::Replaced(replaced)) = told.event else {
        panic!("a replacement: {told:?}");
    };
    assert!(!replaced.state_carried);
    let anew = store.by_key(URL).expect("the new row");
    assert_eq!(anew.status, WorkerStatus::Ready);
    let drain = proto::DeregisterWorkerRequest {
        worker_id: anew.id.0,
        revision: anew.revision,
        drain: true,
    };
    door.deregister(&operator(), under(TERM), &drain)
        .expect("draining again");
    // The URL's next incarnation is another registration: the drain is
    // not handed on, and an unchecked worker is ready from its own.
    let told = door
        .register(
            &operator(),
            under(TERM),
            &ask(unchecked(URL), Mode::Replace),
        )
        .expect("replaced");
    let Some(Event::Replaced(replaced)) = told.event else {
        panic!("a replacement: {told:?}");
    };
    assert!(!replaced.state_carried);
    assert_eq!(
        store.by_key(URL).map(|row| row.status),
        Some(WorkerStatus::Ready)
    );
}

#[test]
fn a_rows_revision_counts_its_stated_facts_and_no_status_change() {
    let (store, door) = door();
    // An unchecked worker turns ready behind its registration's answer
    // and is drained: neither moves the revision that answer gave.
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Add))
            .expect("registered"),
    );
    let drain = proto::DeregisterWorkerRequest {
        worker_id: row.id,
        revision: row.revision,
        drain: true,
    };
    door.deregister(&operator(), under(TERM), &drain)
        .expect("draining");
    assert_eq!(
        store.by_key(URL).map(|row| (row.status, row.revision)),
        Some((WorkerStatus::Draining, row.revision))
    );
    // A patch states a fact: the revision moves, the one before it is
    // refused and the new one removes the row.
    let patch = proto::PatchWorkerRequest {
        worker_id: row.id,
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        revision: row.revision,
        ..Default::default()
    };
    let patched = row_of(
        &door
            .patch(&operator(), under(TERM), &patch)
            .expect("patched"),
    );
    assert_eq!(patched.revision, row.revision + 1);
    let stale = proto::DeregisterWorkerRequest {
        drain: false,
        ..drain
    };
    let refused = door
        .deregister(&operator(), under(TERM), &stale)
        .expect_err("the facts moved");
    assert_eq!(refused.code(), Code::FailedPrecondition);
    let removal = proto::DeregisterWorkerRequest {
        revision: patched.revision,
        ..stale
    };
    door.deregister(&operator(), under(TERM), &removal)
        .expect("removed");
}
