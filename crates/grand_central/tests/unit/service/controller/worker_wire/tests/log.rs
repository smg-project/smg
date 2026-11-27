//! The log's entries, a page of rows and the membership events: what each
//! goes out as on the wire, and what each comes back as.

use super::*;

#[test]
fn a_log_entry_of_each_membership_write_has_its_case_at_the_entrys_place() {
    let old = full();
    let new = WorkerRow {
        id: identity(URL, 4),
        status: WorkerStatus::Ready,
        revision: old.revision + 3,
        ..full()
    };
    let at = Position { seq: 9, term: 4 };
    let entry = |change, row: Option<&WorkerRow>, left: Option<&WorkerRow>| Entry {
        at,
        event: Written::Member {
            epoch: 6,
            change,
            row: row.cloned().map(Arc::new),
            left: left.cloned().map(Arc::new),
        },
    };
    let told = |entry: &Entry| {
        let sent = member(entry).expect("a membership event");
        assert_eq!((sent.seq, sent.membership_epoch, sent.term), (9, 6, 4));
        sent.event.expect("a case")
    };

    let registered = MembershipChange::Registered { id: new.id };
    assert_eq!(
        told(&entry(registered, Some(&new), None)),
        Event::Registered(row(&new))
    );
    for change in [
        MembershipChange::Relabelled { id: new.id },
        MembershipChange::Resized { id: new.id },
    ] {
        assert_eq!(
            told(&entry(change, Some(&new), None)),
            Event::Relabelled(row(&new))
        );
    }
    // The new incarnation's status came over from the old one's, or it
    // is a fresh row's.
    let replacement = MembershipChange::Replaced {
        old: old.id,
        new: new.id,
    };
    assert_eq!(
        told(&entry(replacement, Some(&new), Some(&old))),
        replaced(&old, &new, true)
    );
    let fresh = WorkerRow {
        status: WorkerStatus::Pending,
        ..new.clone()
    };
    assert_eq!(
        told(&entry(replacement, Some(&fresh), Some(&old))),
        replaced(&old, &fresh, false)
    );
    assert_eq!(
        told(&entry(
            MembershipChange::Removed { id: old.id },
            None,
            Some(&old)
        )),
        removed(&old)
    );
    // A status entry is read by its kind. It holds no row, and its
    // event is the one a status write was always sent as, field by
    // field: the models it holds are for a watch's filter alone.
    let (from, to) = (WorkerStatus::Pending, WorkerStatus::Ready);
    let moved = Entry {
        at,
        event: Written::Status {
            epoch: 6,
            id: new.id,
            from,
            to,
            models: new.facts.models.as_slice().into(),
        },
    };
    assert_eq!(told(&moved), status_changed(new.id, from, to));
    assert_eq!(
        member(&moved),
        Some(proto::WorkerEvent {
            seq: 9,
            membership_epoch: 6,
            event: Some(Event::Status(proto::StatusChanged {
                worker_id: new.id.0,
                old: proto::WorkerStatus::Pending as i32,
                new: proto::WorkerStatus::Ready as i32,
                health: proto::HealthFlag::Healthy as u32,
            })),
            term: 4,
        })
    );
    // A shape the store never tells, and an entry that is no worker's
    // event. A status is none of a membership entry's shapes, with a
    // row or with none.
    assert_eq!(member(&entry(registered, None, None)), None);
    let status = MembershipChange::StatusChanged {
        id: new.id,
        from,
        to,
    };
    assert_eq!(member(&entry(status, Some(&new), None)), None);
    assert_eq!(member(&entry(status, None, Some(&old))), None);
    assert_eq!(
        member(&Entry {
            at,
            event: Written::Term
        }),
        None
    );
}

#[test]
fn a_page_carries_its_rows_at_the_cuts_place_and_the_word_to_start_again_carries_none() {
    let first = full();
    let second = WorkerRow {
        id: identity(URL, 2),
        ..full()
    };
    let at = Position { seq: 9, term: 4 };
    assert_eq!(
        page(vec![row(&first), row(&second)], false, at, 6),
        proto::WorkerEvent {
            seq: 9,
            membership_epoch: 6,
            event: Some(Event::Snapshot(proto::WorkerSnapshot {
                workers: vec![row(&first), row(&second)],
                last: false,
            })),
            term: 4,
        }
    );
    assert_eq!(
        page(Vec::new(), true, at, 6).event,
        Some(Event::Snapshot(proto::WorkerSnapshot {
            workers: Vec::new(),
            last: true,
        }))
    );
    // The oldest event the log retains is the one after its floor.
    assert_eq!(
        start_again(Position { seq: 40, term: 4 }),
        proto::WorkerEvent {
            seq: 0,
            membership_epoch: 0,
            event: Some(Event::SnapshotRequired(proto::SnapshotRequired {
                oldest_seq: 41,
            })),
            term: 0,
        }
    );
}

#[test]
fn every_kind_of_log_entry_is_sent_under_a_case_of_its_own() {
    let at = Position { seq: 3, term: 2 };
    let stored = full();
    let entries = [
        Written::Term,
        Written::Member {
            epoch: 1,
            change: MembershipChange::Registered { id: stored.id },
            row: Some(Arc::new(stored.clone())),
            left: None,
        },
        Written::Status {
            epoch: 1,
            id: stored.id,
            from: WorkerStatus::Pending,
            to: WorkerStatus::Ready,
            models: stored.facts.models.as_slice().into(),
        },
    ]
    .map(|event| replication(&Entry { at, event }).event);
    assert_eq!(entries[0], Some(Sent::TermStart(proto::TermStart {})));
    assert_eq!(
        entries[1],
        Some(Sent::Membership(event(
            Event::Registered(row(&stored)),
            3,
            1,
            2
        )))
    );
    // A status entry is sent as the status event of its change: the
    // models it holds are not sent.
    let moved = status_changed(stored.id, WorkerStatus::Pending, WorkerStatus::Ready);
    assert_eq!(entries[2], Some(Sent::Membership(event(moved, 3, 1, 2))));
    // A membership entry in a shape the store never tells is sent with
    // no case set: the standby reads an event it does not know.
    let untold = Written::Member {
        epoch: 1,
        change: MembershipChange::Registered { id: stored.id },
        row: None,
        left: None,
    };
    assert_eq!(replication(&Entry { at, event: untold }).event, None);
}

#[test]
fn every_membership_event_comes_back_as_the_change_the_store_follows() {
    let old = full();
    let new = WorkerRow {
        id: identity(URL, 4),
        status: WorkerStatus::Ready,
        ..full()
    };
    let now = new.admitted_at;
    let sent = |change, row: Option<&WorkerRow>, left: Option<&WorkerRow>| {
        let entry = Entry {
            at: Position { seq: 9, term: 4 },
            event: Written::Member {
                epoch: 6,
                change,
                row: row.cloned().map(Arc::new),
                left: left.cloned().map(Arc::new),
            },
        };
        told_change(&member(&entry).expect("a membership event"), now)
    };
    let registered = MembershipChange::Registered { id: new.id };
    let relabelled = MembershipChange::Relabelled { id: new.id };
    let resized = MembershipChange::Resized { id: new.id };
    let replacement = MembershipChange::Replaced {
        old: old.id,
        new: new.id,
    };
    let removal = MembershipChange::Removed { id: old.id };
    let status = MembershipChange::StatusChanged {
        id: old.id,
        from: WorkerStatus::Ready,
        to: WorkerStatus::Draining,
    };
    let row = Some(new.clone());
    assert_eq!(
        sent(registered, Some(&new), None),
        Ok((6, registered, row.clone()))
    );
    assert_eq!(
        sent(relabelled, Some(&new), None),
        Ok((6, relabelled, row.clone()))
    );
    // A resize crosses as a relabel: the store reads the widths.
    assert_eq!(
        sent(resized, Some(&new), None),
        Ok((6, relabelled, row.clone()))
    );
    assert_eq!(
        sent(replacement, Some(&new), Some(&old)),
        Ok((6, replacement, row))
    );
    assert_eq!(sent(removal, None, Some(&old)), Ok((6, removal, None)));
    // A status entry comes back as the status change, with no row.
    let moved = Entry {
        at: Position { seq: 9, term: 4 },
        event: Written::Status {
            epoch: 6,
            id: old.id,
            from: WorkerStatus::Ready,
            to: WorkerStatus::Draining,
            models: old.facts.models.as_slice().into(),
        },
    };
    let back = told_change(&member(&moved).expect("a status event"), now);
    assert_eq!(back, Ok((6, status, None)));
    // A watch's own cases and an event of no case are not a change.
    for event in [
        None,
        Some(Event::Snapshot(proto::WorkerSnapshot::default())),
    ] {
        let none = proto::WorkerEvent {
            event,
            ..proto::WorkerEvent::default()
        };
        let refused = told_change(&none, now).expect_err("no change");
        assert_eq!(refused.path, "membership.event");
    }
}
