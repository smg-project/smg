//! The cases of a registration's checks and its connect signal: a row whose
//! checks are off is sent the signal at once, a registration that states no
//! policy is checked, and whether a registration sent again sends the signal.

use super::*;

#[test]
fn a_registration_with_its_checks_off_is_ready_at_once() {
    let (store, door) = door();
    // Not waited for: the answer is the pending row, and the store has
    // moved it to ready by the next read, at the revision the answer
    // gave: a status change moves none.
    let told = door
        .register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Add))
        .expect("registered");
    let row = row_of(&told);
    assert_eq!(
        (row.status, row.revision, told.membership_epoch),
        (proto::WorkerStatus::Pending as i32, 1, 1)
    );
    let stored = store.get(numbered(URL, 0)).expect("the row");
    assert_eq!(
        (stored.status, stored.revision, store.epoch()),
        (WorkerStatus::Ready, 1, 2)
    );

    // Waited for: the answer is the row after its connect signal.
    let other = "grpc://10.0.0.6:8000";
    let waited = proto::RegisterWorkerRequest {
        wait_ready: true,
        ..ask(unchecked(other), Mode::Add)
    };
    let told = door
        .register(&operator(), under(TERM), &waited)
        .expect("registered");
    let row = row_of(&told);
    assert_eq!(
        (row.status, row.revision, told.membership_epoch),
        (proto::WorkerStatus::Ready as i32, 1, 4)
    );
    assert_eq!(row.health, proto::HealthFlag::Healthy as u32);

    // A checked row that turns its checks off is ready from that write.
    let third = "grpc://10.0.0.7:8000";
    door.register(&operator(), under(TERM), &ask(spec(third), Mode::Add))
        .expect("registered");
    door.register(
        &operator(),
        under(TERM),
        &ask(unchecked(third), Mode::Upsert),
    )
    .expect("relabelled");
    let stored = store.by_key(third).expect("the row");
    assert_eq!(stored.status, WorkerStatus::Ready);
}

/// A call that does not wait answers the row as it stood, and the
/// signal it sent again readies the row behind its answer, as a
/// registration that wrote answers its row ahead of the signal. The
/// registration is sent again as an upsert: the same add is refused
/// where its row stands, as any add is, and sends no signal.
#[test]
fn an_equal_registration_that_does_not_wait_answers_the_row_ahead_of_its_signal() {
    let (store, door, _) = lost_its_lead_at(0, &ask(unchecked(URL), Mode::Add));
    store::LeasesStore::set_term(store.as_ref(), TERM + 2);
    let stood = store.by_key(URL).expect("the row");
    let refused = door
        .register(
            &operator(),
            under(TERM + 2),
            &ask(unchecked(URL), Mode::Add),
        )
        .expect_err("the row stands");
    assert_eq!(refused.code(), Code::AlreadyExists);
    assert_eq!(store.get(stood.id).as_ref(), Some(&stood));
    let told = door
        .register(
            &operator(),
            under(TERM + 2),
            &ask(unchecked(URL), Mode::Upsert),
        )
        .expect("unmoved");
    assert_eq!(
        told.event,
        Some(Event::Relabelled(super::super::worker_wire::row(&stood)))
    );
    assert_eq!(told.membership_epoch, 1);
    let ready = store.get(stood.id).expect("the row");
    assert_eq!(
        (ready.status, ready.revision, store.epoch()),
        (WorkerStatus::Ready, stood.revision, 2)
    );
}

/// The host door's routine sends the signal again too, for a rank
/// that stands unmoved. This add of three ranks, their checks off,
/// lost its lead as its first rank's write landed: the rank stands
/// pending, its signal not written, and the call was refused at the
/// next. The same add called again lays the two ranks left and sends
/// the first its signal, which asks its own term: a call admitted under
/// the term that ended sends nothing. It waits, so it answers each
/// rank behind its signal, the unmoved one too. An add writes only
/// where a rank is left to write: the next case has the engine whole.
#[test]
fn an_add_called_again_sends_an_unmoved_ranks_connect_signal_again() {
    use proto::rank_row::Made;
    let ranks = || proto::WorkerSpec {
        dp_size: 3,
        ..unchecked(URL)
    };
    let (store, door, lost) = lost_its_lead_at(0, &ask(ranks(), Mode::Add));
    let lost = lost.expect_err("the lead was lost at the second rank");
    assert_eq!(lost.code(), Code::Unavailable);
    store::LeasesStore::set_term(store.as_ref(), TERM + 2);
    let first = store.by_key(&format!("{URL}@0")).expect("the first rank");
    assert_eq!(
        (first.status, store.epoch(), store.all_rows().1.len()),
        (WorkerStatus::Pending, 1, 1)
    );
    let again = proto::RegisterWorkerRequest {
        wait_ready: true,
        ..ask(ranks(), Mode::Add)
    };
    // The log leads a later term. A call admitted under the one that
    // ended is refused ahead of its first rank, and sends nothing.
    let ended = door
        .register(&operator(), under(TERM), &again)
        .expect_err("its term ended");
    assert_eq!(ended.code(), Code::Unavailable);
    assert_eq!(store.get(first.id).as_ref(), Some(&first));
    let told = door
        .register(&operator(), under(TERM + 2), &again)
        .expect("the same add");
    let made = made_of(&told);
    let ready = proto::WorkerStatus::Ready as i32;
    assert!(
        matches!(
            made.as_slice(),
            [Made::Relabelled(stood), Made::Registered(second), Made::Registered(third)]
                if (stood.id, stood.revision) == (first.id.0, first.revision)
                    && [stood.status, second.status, third.status] == [ready; 3]
        ),
        "{made:?}"
    );
    let (_, rows) = store.all_rows();
    assert_eq!(rows.len(), 3);
    for row in rows {
        assert_eq!(row.status, WorkerStatus::Ready, "{}", row.facts.key);
    }
    assert_eq!(
        store.get(first.id).map(|row| row.revision),
        Some(first.revision)
    );
}

/// An add sent again over an engine that stands whole is refused ahead
/// of every write and sends no signal, as an add over one row is: an
/// add changes no row that stands. This add of two ranks, their checks
/// off, lost its lead as its last rank's write landed, so both ranks
/// stand and the last waits for its signal. The same add is refused
/// and the rank waits on. An upsert of equal facts sends the signal.
/// It does not wait, so it answers the rank as it stood, ahead of the
/// signal.
#[test]
fn an_add_over_an_engine_that_stands_whole_sends_no_signal_and_an_upsert_sends_it() {
    use proto::rank_row::Made;
    let ranks = || proto::WorkerSpec {
        dp_size: 2,
        ..unchecked(URL)
    };
    // The lead holds for the first rank's row and its signal, and is
    // lost as the last rank's row lands.
    let (store, door, told) = lost_its_lead_at(2, &ask(ranks(), Mode::Add));
    assert_eq!(made_of(&told.expect("both ranks registered")).len(), 2);
    let status = |rank: u32| {
        let row = store.by_key(&format!("{URL}@{rank}"));
        row.map(|row| row.status)
    };
    let (ready, pending) = (Some(WorkerStatus::Ready), Some(WorkerStatus::Pending));
    assert_eq!((status(0), status(1), store.epoch()), (ready, pending, 3));
    store::LeasesStore::set_term(store.as_ref(), TERM + 2);
    let whole = door
        .register(&operator(), under(TERM + 2), &ask(ranks(), Mode::Add))
        .expect_err("the engine stands whole");
    assert_eq!(whole.code(), Code::AlreadyExists);
    assert_eq!((status(1), store.epoch()), (pending, 3));
    let waiting = store.by_key(&format!("{URL}@1")).expect("the last rank");
    let told = door
        .register(&operator(), under(TERM + 2), &ask(ranks(), Mode::Upsert))
        .expect("unmoved");
    let made = made_of(&told);
    assert!(
        matches!(
            made.as_slice(),
            [Made::Relabelled(first), Made::Relabelled(last)]
                if first.status == proto::WorkerStatus::Ready as i32
                    && *last == super::super::worker_wire::row(&waiting)
        ),
        "{made:?}"
    );
    assert_eq!((status(0), status(1), store.epoch()), (ready, ready, 4));
    assert_eq!(
        store.get(waiting.id).map(|row| row.revision),
        Some(waiting.revision)
    );
}

/// A checked row takes no connect signal from equal facts, pending as
/// it stands: a prober's verdict readies it, and the registration sent
/// again writes nothing. So it is for a call of one row and for each
/// rank of a wide call, which the host door's routine lays.
#[test]
fn a_pending_row_whose_checks_are_on_takes_no_signal_from_equal_facts() {
    let (store, door) = door();
    for stated in [spec(URL), wide("grpc://10.0.0.6:8000", 2)] {
        door.register(&operator(), under(TERM), &ask(stated.clone(), Mode::Add))
            .expect("registered");
        let epoch = store.epoch();
        door.register(&operator(), under(TERM), &ask(stated, Mode::Upsert))
            .expect("unmoved");
        assert_eq!(store.epoch(), epoch);
    }
    let (_, rows) = store.all_rows();
    assert_eq!(rows.len(), 3);
    for row in rows {
        let held = door.row(row.id).map(|row| row.status);
        assert_eq!(
            (row.status, held),
            (WorkerStatus::Pending, Some(WorkerStatus::Pending)),
            "{}",
            row.facts.key
        );
    }
}

#[test]
fn a_registration_that_states_no_policy_is_checked_whatever_stood_under_its_url() {
    let (store, door) = door();
    let status = |url: &str| store.by_key(url).map(|row| row.status);
    door.register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Add))
        .expect("registered");
    assert_eq!(status(URL), Some(WorkerStatus::Ready));
    // The URL's next worker (another model: no status comes over)
    // states no policy. Nobody said "do not check" of it: a wait for
    // its handshake is refused as any checked worker's is, and it is
    // pending until a verdict.
    let other = proto::WorkerSpec {
        models: vec!["model-b".to_owned()],
        ..spec(URL)
    };
    let waited = proto::RegisterWorkerRequest {
        wait_ready: true,
        ..ask(other.clone(), Mode::Replace)
    };
    let refused = door
        .register(&operator(), under(TERM), &waited)
        .expect_err("no handshake runs");
    assert_eq!(refused.code(), Code::InvalidArgument);
    let told = door
        .register(&operator(), under(TERM), &ask(other, Mode::Replace))
        .expect("replaced");
    let row = row_of(&told);
    assert_eq!(
        (row.status, row.health_policy.map(|policy| policy.disabled)),
        (proto::WorkerStatus::Pending as i32, Some(false))
    );
    assert_eq!(status(URL), Some(WorkerStatus::Pending));

    // In place too: an upsert that states no policy over an unchecked
    // row turns its checks on and leaves its status to the prober.
    let third = "grpc://10.0.0.7:8000";
    door.register(&operator(), under(TERM), &ask(unchecked(third), Mode::Add))
        .expect("registered");
    let told = door
        .register(&operator(), under(TERM), &ask(spec(third), Mode::Upsert))
        .expect("in place");
    assert!(matches!(told.event, Some(Event::Relabelled(_))));
    let row = row_of(&told);
    assert_eq!(
        (row.status, row.health_policy.map(|policy| policy.disabled)),
        (proto::WorkerStatus::Ready as i32, Some(false))
    );
}
