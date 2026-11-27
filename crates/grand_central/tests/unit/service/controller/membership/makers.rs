//! The cases of a row's maker: the maker a registration gives its row, what
//! a caller that did not make a row changes of it, the room a maker takes
//! in a row, the guarded row a provider takes as its maker, and an add of
//! one row over a row that stands, refused whoever made it.

use super::*;

#[test]
fn a_row_keeps_the_maker_its_registration_gave_it() {
    let (store, door) = door();
    // A control call's row says no maker on the wire.
    let by_hand = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
            .expect("registered"),
    );
    assert_eq!(by_hand.maker, None);
    let held = store.get(WorkerId(by_hand.id)).expect("the row");
    assert_eq!(held.maker, Registrant::Control);
    // A provider's registration makes the row its own, and the answer
    // shows whose it is.
    let maker = found("pods/a");
    let other = "grpc://10.0.0.6:8000";
    let made = row_of(
        &door
            .upsert(&operator(), &maker, TERM, &ask(spec(other), Mode::Upsert))
            .expect("found"),
    );
    let shown = Some(proto::Maker {
        provider: "cluster".to_owned(),
        key: "pods/a".to_owned(),
    });
    assert_eq!(made.maker, shown);
    // Its refresh in place and its new incarnation keep the maker. The
    // refresh comes from a maker made anew: two makers are one where
    // they agree whole, whichever copy a caller holds.
    let relabelled = proto::WorkerSpec {
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        ..spec(other)
    };
    let anew = found("pods/a");
    let in_place = row_of(
        &door
            .upsert(&operator(), &anew, TERM, &ask(relabelled, Mode::Upsert))
            .expect("refreshed"),
    );
    assert_eq!((in_place.id, &in_place.maker), (made.id, &shown));
    let again = row_of(
        &door
            .upsert(&operator(), &maker, TERM, &ask(spec(other), Mode::Replace))
            .expect("replaced"),
    );
    assert_ne!(again.id, made.id);
    assert_eq!(again.maker, shown);
    let held = store.get(WorkerId(again.id)).expect("the row");
    assert_eq!(held.maker, maker);
}

#[test]
fn a_caller_that_did_not_make_a_row_changes_nothing_of_it() {
    let (store, door) = door();
    let maker = found("pods/a");
    let made = row_of(
        &door
            .upsert(&operator(), &maker, TERM, &ask(spec(URL), Mode::Upsert))
            .expect("found"),
    );
    let epoch = store.epoch();
    let refused = |answer: Result<proto::WorkerEvent, Status>, whose: &str| {
        let status = answer.expect_err("refused");
        assert_eq!(status.code(), Code::PermissionDenied, "{status:?}");
        assert!(status.message().contains(whose), "{status:?}");
        assert!(!status.message().contains("10.0.0"), "no URL: {status:?}");
    };
    // A control call on the provider's row: a registration over it, with
    // other facts and with equal ones, a patch, a drain and a removal.
    let relabelled = proto::WorkerSpec {
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        ..spec(URL)
    };
    for request in [
        ask(spec(URL), Mode::Replace),
        ask(relabelled, Mode::Upsert),
        ask(spec(URL), Mode::Upsert),
    ] {
        let answer = door.register(&operator(), under(TERM), &request);
        refused(answer, "provider cluster");
    }
    let patch = proto::PatchWorkerRequest {
        worker_id: made.id,
        labels: [("lane".to_owned(), "b".to_owned())].into(),
        remove_labels: Vec::new(),
        overload: None,
        revision: made.revision,
    };
    refused(
        door.patch(&operator(), under(TERM), &patch),
        "provider cluster",
    );
    for drain in [true, false] {
        let leave = proto::DeregisterWorkerRequest {
            worker_id: made.id,
            revision: made.revision,
            drain,
        };
        refused(
            door.deregister(&operator(), under(TERM), &leave),
            "provider cluster",
        );
    }
    // The maker is tested ahead of the revision: a caller that did not
    // make the row is told so whatever revision it read.
    let stale = proto::PatchWorkerRequest {
        revision: made.revision + 7,
        ..patch.clone()
    };
    refused(
        door.patch(&operator(), under(TERM), &stale),
        "provider cluster",
    );
    // The same provider under another key is another worker's maker.
    let another = found("pods/b");
    refused(
        door.upsert(&operator(), &another, TERM, &ask(spec(URL), Mode::Upsert)),
        "provider cluster",
    );
    // Nothing was written, and the maker's own call goes through.
    assert_eq!(store.epoch(), epoch);
    let held = store.get(WorkerId(made.id)).expect("the row");
    assert_eq!(held.revision, made.revision);
    door.relabel(&operator(), &maker, TERM, &patch)
        .expect("the maker patches its row");

    // The other way round: a provider changes nothing of a control
    // call's row.
    let other = "grpc://10.0.0.6:8000";
    let by_hand = row_of(
        &door
            .register(&operator(), under(TERM), &ask(spec(other), Mode::Add))
            .expect("registered"),
    );
    refused(
        door.upsert(&operator(), &maker, TERM, &ask(spec(other), Mode::Upsert)),
        "a control call",
    );
    let leave = proto::DeregisterWorkerRequest {
        worker_id: by_hand.id,
        revision: by_hand.revision,
        drain: false,
    };
    refused(
        door.remove(&operator(), &maker, TERM, &leave),
        "a control call",
    );
    door.deregister(&operator(), under(TERM), &leave)
        .expect("the control call removes its row");
}

#[test]
fn a_row_holds_its_maker_inside_the_wires_bound() {
    use super::super::worker_wire;

    let (store, door) = door();
    let maker = found("pods/a");
    // Three labels at a value's bound and one that fills. The widest
    // fill a row of this maker has room for is registered; one byte more
    // is refused by the row's name, and so is a patch of the row at its
    // brim.
    let stated = |fill: usize| proto::WorkerSpec {
        labels: ["a", "b", "c"]
            .map(|key| (key.to_owned(), "v".repeat(4_096)))
            .into_iter()
            .chain([("fill".to_owned(), "v".repeat(fill))])
            .collect(),
        ..spec(URL)
    };
    let fits = |fill: usize| {
        worker_wire::registration(&stated(fill), |_| None).is_ok_and(|(facts, _)| {
            worker_wire::room(&facts, &facts.labels, &maker, "spec").is_ok()
        })
    };
    let widest = (0..=4_096usize)
        .rev()
        .find(|&fill| fits(fill))
        .expect("a fill the row has room for");
    let epoch = store.epoch();
    let past = door.upsert(
        &operator(),
        &maker,
        TERM,
        &ask(stated(widest + 1), Mode::Upsert),
    );
    let status = past.expect_err("past the bound");
    assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
    assert!(status.message().starts_with("spec:"), "{status:?}");
    assert_eq!(store.epoch(), epoch);
    let brim = row_of(
        &door
            .upsert(
                &operator(),
                &maker,
                TERM,
                &ask(stated(widest), Mode::Upsert),
            )
            .expect("at the bound"),
    );
    let more = proto::PatchWorkerRequest {
        worker_id: brim.id,
        labels: [("lane".to_owned(), "b".to_owned())].into(),
        remove_labels: Vec::new(),
        overload: None,
        revision: brim.revision,
    };
    let status = door
        .relabel(&operator(), &maker, TERM, &more)
        .expect_err("past the bound");
    assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
    assert!(status.message().starts_with("labels:"), "{status:?}");
}

#[test]
fn a_provider_acting_as_maker_takes_only_its_own_guarded_row() {
    let (store, door) = door();
    let door = Arc::new(door);
    let hosted = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM));
    let maker = found("pods/a");
    hosted
        .found_one(&maker, URL, &|_| found_facts())
        .expect("found");
    let made = store.by_key(URL).expect("the found row");
    let other = "grpc://10.0.0.6:8000";
    door.register(&operator(), under(TERM), &ask(spec(other), Mode::Add))
        .expect("registered");
    let epoch = store.epoch();
    let denied = |answer: Result<MembershipWriteOutcome, MembershipWriteRefused>, whose: &str| {
        let Err(MembershipWriteRefused::NotRegistrant(why)) = answer else {
            panic!("wrong refusal: {answer:?}")
        };
        assert!(why.contains(whose), "{why}");
        assert!(!why.contains("10.0.0"), "{why}");
        assert!(!why.contains("pods/"), "{why}");
    };
    // Another maker's row stays: a control call's, and this provider's
    // own under another key, which is told so.
    let by = Remover::Registrant(&maker);
    let seen = standing(&made);
    let control = store.by_key(other).expect("the control row");
    denied(hosted.gone(by, other, standing(&control)), "a control call");
    denied(
        hosted.gone(Remover::Registrant(&found("pods/b")), URL, seen),
        "provider cluster's under another key",
    );
    // A caller of another provider is told whose the row is, and
    // nothing of a key.
    let stranger = Registrant::provider("nodes", "pods/a").expect("a whole maker");
    denied(
        hosted.gone(Remover::Registrant(&stranger), URL, seen),
        "provider cluster's: its worker",
    );
    // No row under the URL is not found, and a replica that does not
    // lead refuses by name. None of the five wrote.
    assert_eq!(
        hosted.gone(by, "grpc://10.0.0.9:8000", seen),
        Err(MembershipWriteRefused::Absent)
    );
    let follower = || -> Result<u64, Status> {
        Err(Status::failed_precondition("this replica does not lead"))
    };
    let follower = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), follower);
    assert_eq!(
        follower.gone(by, URL, seen),
        Err(MembershipWriteRefused::NotLeading(
            "this replica does not lead".to_owned()
        ))
    );
    assert_eq!(store.epoch(), epoch);
    // The maker's own call takes the row away and answers it removed.
    assert_eq!(
        hosted.gone(by, URL, seen),
        Ok(MembershipWriteOutcome::Removed(made.id))
    );
    assert!(store.get(made.id).is_none());
    assert_eq!(store.epoch(), epoch + 1);
    // The old guard still names the previous row at this URL.
    hosted
        .found_one(&found("pods/c"), URL, &|_| found_facts())
        .expect("found again");
    let again = store.by_key(URL).expect("the new row");
    assert_ne!(again.id, made.id);
    assert_eq!(
        hosted.gone(by, URL, seen),
        Err(MembershipWriteRefused::Moved)
    );
    assert_eq!(store.by_key(URL), Some(again));
}

#[test]
fn an_add_over_a_standing_row_is_already_there_whoever_made_it() {
    let (store, door) = door();
    let maker = found("pods/a");
    door.upsert(&operator(), &maker, TERM, &ask(spec(URL), Mode::Upsert))
        .expect("found");
    let other = "grpc://10.0.0.6:8000";
    door.register(&operator(), under(TERM), &ask(spec(other), Mode::Add))
        .expect("registered");
    let epoch = store.epoch();
    let there = |answer: Result<proto::WorkerEvent, Status>| {
        let status = answer.expect_err("refused");
        assert_eq!(status.code(), Code::AlreadyExists, "{status:?}");
    };
    // An add asks no maker: a control call's and a provider's read the
    // same over a provider's row and over a control call's.
    for url in [URL, other] {
        there(door.register(&operator(), under(TERM), &ask(spec(url), Mode::Add)));
        there(door.upsert(&operator(), &maker, TERM, &ask(spec(url), Mode::Add)));
    }
    assert_eq!(store.epoch(), epoch);
}
