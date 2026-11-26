//! The operators' snapshot: what the leader serves, what it refuses, and
//! how a snapshot takes its turn among the feed's readers.

use super::*;

/// A `GetSnapshot` call from `operator-a` for `slices`.
fn snapshot(slices: &[proto::snapshot_request::Slice]) -> Request<proto::SnapshotRequest> {
    from(
        "operator-a",
        proto::SnapshotRequest {
            slices: slices.iter().map(|slice| i32::from(*slice)).collect(),
            ..Default::default()
        },
    )
}

/// The operators' snapshot on the leader, under a spec store: the
/// header, the rows as an operator reads them and the spec's status. A
/// row registered under a URL with userinfo is told by the id and the
/// revision a patch names, its address without the userinfo. The status
/// lists the generation a rollback can name. A slice that was not asked
/// for is absent, and the answer names no place of the log.
#[tokio::test]
async fn the_leader_serves_the_snapshot_of_its_rows_and_its_spec() {
    use proto::snapshot_request::Slice;

    let (_dir, _elector, _standing, control) = stored(true);
    for document in [LEAST_LOAD, RANDOM] {
        let ran = control
            .apply_policy_spec(apply(proto::ApplyMode::Enforce, document, "yaml", ""))
            .await
            .expect("a verdict")
            .into_inner();
        assert!(ran.accepted, "{ran:?}");
    }
    let url = "grpc://operator-name:s3cret-word@10.0.0.5:8000";
    let spec = proto::WorkerSpec {
        url: url.to_owned(),
        wire: proto::WireMode::Grpc as i32,
        role: proto::Role::Single as i32,
        models: vec!["model-a".to_owned()],
        ..Default::default()
    };
    let request = proto::RegisterWorkerRequest {
        spec: Some(spec),
        ..Default::default()
    };
    let registered = control
        .register_worker(from("operator-a", request))
        .await
        .expect("registered")
        .into_inner();
    let Some(proto::worker_event::Event::Registered(stated)) = registered.event else {
        panic!("a registration's event: {registered:?}");
    };
    // The registration's answer tells its caller the URL it stated.
    assert_eq!(stated.url, url);

    let answer = control
        .get_snapshot(snapshot(&[Slice::Workers, Slice::Spec]))
        .await
        .expect("a snapshot")
        .into_inner();
    assert_eq!(
        (
            answer.term,
            answer.generation,
            answer.controller_id.as_str()
        ),
        (1, 2, ID)
    );
    let [row] = answer.workers.as_slice() else {
        panic!("one row: {answer:?}");
    };
    // The snapshot tells the row to any operator: by what a patch names
    // it, and without the userinfo.
    assert_eq!((row.id, row.revision), (stated.id, stated.revision));
    let clean = "grpc://10.0.0.5:8000";
    assert_eq!(
        (
            row.url.as_str(),
            row.registration_key.as_str(),
            row.engine.as_str()
        ),
        (clean, clean, clean)
    );
    assert!(!answer.truncated);
    let status = answer.spec.expect("the spec slice");
    let standing = control.shared.generations.standing();
    assert_eq!(status.generation, 2);
    assert_eq!(status.digest, standing.1.expect("a digest").to_string());
    assert_eq!(status.retained, [1]);
    assert_eq!(status.source, "api");
    assert_eq!(status.mode, proto::ApplyMode::Enforce as i32);
    assert!(status.applied_unix_micros > 0, "{status:?}");
    // Redacted, and no base a standby starts from.
    assert_eq!(
        (
            answer.replication_seq,
            answer.replication_term,
            answer.membership_epoch
        ),
        (0, 0, 0)
    );
    assert!(answer.ids.is_none() && answer.documents.is_empty());

    // One slice asked for: the other is absent.
    let rows = control
        .get_snapshot(snapshot(&[Slice::Workers]))
        .await
        .expect("the rows")
        .into_inner();
    assert_eq!(
        (rows.workers.len(), rows.spec, rows.generation),
        (1, None, 2)
    );
    let status = control
        .get_snapshot(snapshot(&[Slice::Spec]))
        .await
        .expect("the status")
        .into_inner();
    assert!(status.workers.is_empty() && status.spec.is_some());
}

/// A replica that does not lead serves no snapshot: it refuses as it
/// refuses a watch's open, with the `NotLeader` that names the holder,
/// before it reads the request's fields.
#[tokio::test]
async fn a_replica_that_does_not_lead_serves_no_snapshot() {
    use proto::snapshot_request::Slice;

    let (_elector, _standing, control) = sourced(false);
    for request in [
        snapshot(&[Slice::Workers, Slice::Spec]),
        // A request no leader serves either: the lead is read first.
        from("operator-a", proto::SnapshotRequest::default()),
    ] {
        let refused = control.get_snapshot(request).await.expect_err("refused");
        assert_eq!(
            (refused.code(), refused.message()),
            (Code::Unavailable, super::super::FOLLOWS_NO_LOG)
        );
        let details = RpcStatus::decode(refused.details()).expect("a google.rpc.Status");
        let [detail] = details.details.as_slice() else {
            panic!("one detail: {details:?}");
        };
        let named = proto::NotLeader::decode(detail.value.as_slice()).expect("a NotLeader");
        assert_eq!(named.leader_address, "10.0.0.8:30100");
    }
}

/// On the leader a snapshot is refused by its request's fields: one
/// that names no slice, a slice this build does not serve, identifiers
/// in clear. A leader with no spec source serves its rows, and refuses
/// the spec slice by its name.
#[tokio::test]
async fn the_leader_refuses_a_snapshot_by_its_requests_field() {
    use proto::snapshot_request::Slice;

    let (_elector, _standing, control) = electing(true);
    let every = from("operator-a", proto::SnapshotRequest::default());
    let refused = control.get_snapshot(every).await.expect_err("refused");
    assert_eq!(refused.code(), Code::Unimplemented);
    assert!(
        refused
            .message()
            .starts_with("slices: none named is every slice"),
        "{}",
        refused.message()
    );
    let health = control
        .get_snapshot(snapshot(&[Slice::Workers, Slice::Health]))
        .await
        .expect_err("refused");
    assert_eq!(
        (health.code(), health.message()),
        (
            Code::Unimplemented,
            "slices: this build does not serve SLICE_HEALTH: it serves SLICE_WORKERS, SLICE_SPEC"
        )
    );
    let plain = proto::SnapshotRequest {
        plain_identifiers: true,
        ..snapshot(&[Slice::Workers]).into_inner()
    };
    let refused = control
        .get_snapshot(from("operator-a", plain))
        .await
        .expect_err("refused");
    assert_eq!(refused.code(), Code::Unimplemented);
    assert!(
        refused.message().starts_with("plain_identifiers: "),
        "{}",
        refused.message()
    );

    // No spec source: the rows alone.
    let spec = control
        .get_snapshot(snapshot(&[Slice::Workers, Slice::Spec]))
        .await
        .expect_err("refused");
    assert_eq!(
        (spec.code(), spec.message()),
        (
            Code::Unimplemented,
            "slices: SLICE_SPEC is not served by a controller started with no spec source"
        )
    );
    let rows = control
        .get_snapshot(snapshot(&[Slice::Workers]))
        .await
        .expect("the rows")
        .into_inner();
    assert_eq!((rows.term, rows.generation, rows.workers.len()), (1, 0, 0));
    assert_eq!(rows.spec, None);
}

/// One snapshot is taken at a time, and no more calls are let in than
/// the turn has places: the call past them is told to ask again. A call
/// refused by the lead or by its request's field takes no place.
#[tokio::test]
async fn a_snapshot_past_the_ones_that_wait_is_told_to_ask_again() {
    use proto::snapshot_request::Slice;

    let (_elector, _standing, control) = electing(true);
    let places = control.shared.readers.snapshots().fill();
    assert_eq!(places.len(), super::super::snapshot::SNAPSHOTS_WAITING + 1);
    let past = control
        .get_snapshot(snapshot(&[Slice::Workers]))
        .await
        .expect_err("no place is left");
    assert_eq!(
        (past.code(), past.message()),
        (
            Code::ResourceExhausted,
            "32 snapshots wait their turn: ask again later"
        )
    );
    // A request refused by its field is refused by it, with no place.
    let health = control
        .get_snapshot(snapshot(&[Slice::Health]))
        .await
        .expect_err("refused");
    assert_eq!(health.code(), Code::Unimplemented);
    // So is the spec slice of a controller with no spec source: the
    // source is read ahead of the turn.
    let spec = control
        .get_snapshot(snapshot(&[Slice::Spec]))
        .await
        .expect_err("refused");
    assert_eq!(
        (spec.code(), spec.message()),
        (
            Code::Unimplemented,
            "slices: SLICE_SPEC is not served by a controller started with no spec source"
        )
    );
    drop(places);
    let rows = control
        .get_snapshot(snapshot(&[Slice::Workers]))
        .await
        .expect("a place is free again");
    assert_eq!(rows.into_inner().term, 1);
    assert_eq!(control.shared.readers.snapshots().let_in(), 0);

    // A replica that does not lead refuses by its lead, places or none.
    let (_elector, _standing, follower) = electing(false);
    let _places = follower.shared.readers.snapshots().fill();
    let refused = follower
        .get_snapshot(snapshot(&[Slice::Workers]))
        .await
        .expect_err("refused");
    assert_eq!(refused.code(), Code::Unavailable);
}

/// A snapshot that states no limit and no models reads the rows as a
/// watch's start does. It shares the cut the feed keeps at the log's
/// head and keeps the one it takes, so calls one behind another copy
/// the rows once a write, and the call behind a write tells it. The
/// last call out lets the cut go where no watch is open: a lone call's
/// cut goes with it, and a call beside another that is let in keeps it
/// for that one.
#[tokio::test]
async fn a_snapshot_shares_the_feeds_cut_and_keeps_none_with_no_watch_open() {
    use proto::snapshot_request::Slice;

    let (_elector, _standing, control) = electing(true);
    let feed = control.shared.feed.clone();
    let told = async || {
        control
            .get_snapshot(snapshot(&[Slice::Workers]))
            .await
            .expect("the rows")
            .into_inner()
            .workers
            .len()
    };
    // No watch is open: the lone call's cut goes with it.
    assert_eq!(told().await, 0);
    assert!(feed.kept().is_none(), "no reader can share the cut");

    // Another call is let in beside it: the cut stays for that one, the
    // next call shares it, and the last call out lets it go.
    let beside = control
        .shared
        .readers
        .snapshots()
        .place()
        .expect("a place is free");
    assert_eq!(told().await, 0);
    let cut = feed.kept().expect("a call is let in to share the cut");
    assert_eq!(told().await, 0);
    assert!(
        Arc::ptr_eq(&cut, &feed.kept().expect("still kept")),
        "one copy for the calls let in"
    );
    drop(beside);
    assert_eq!(told().await, 0);
    assert!(feed.kept().is_none(), "the last call out let it go");

    // A watch is open: the snapshot shares the cut its start took.
    let gateways = Service::new(vec![Scope::Gateway], Arc::clone(&control.shared));
    let mut watching = gateways
        .watch_workers(from("gateway-a", proto::WatchWorkersRequest::default()))
        .await
        .expect("opened")
        .into_inner();
    tokio::time::timeout(Duration::from_secs(5), watching.next())
        .await
        .expect("a page within five seconds")
        .expect("the stream is open")
        .expect("a page, not a status");
    let cut = feed.kept().expect("the watch's cut is kept");
    assert_eq!(told().await, 0);
    assert!(
        Arc::ptr_eq(&cut, &feed.kept().expect("still kept")),
        "the snapshot shared the watch's cut"
    );

    // A write moves the log past that cut: the next snapshot tells the
    // row and keeps the cut it took, and the call behind it shares it.
    let request = proto::RegisterWorkerRequest {
        spec: Some(proto::WorkerSpec {
            url: "grpc://10.0.0.5:8000".to_owned(),
            wire: proto::WireMode::Grpc as i32,
            role: proto::Role::Single as i32,
            models: vec!["model-a".to_owned()],
            ..Default::default()
        }),
        ..Default::default()
    };
    control
        .register_worker(from("operator-a", request))
        .await
        .expect("registered");
    assert_eq!(told().await, 1);
    let later = feed.kept().expect("the snapshot's cut is kept");
    assert!(!Arc::ptr_eq(&cut, &later), "a cut of the later place");
    assert_eq!(told().await, 1);
    assert!(
        Arc::ptr_eq(&later, &feed.kept().expect("still kept")),
        "one copy a write"
    );

    // The last watch leaves, and the next snapshot keeps no cut.
    drop(watching);
    assert_eq!(told().await, 1);
    assert!(feed.kept().is_none(), "no watch can share the cut");
}

/// A lone call with no watch open that states a limit or its models
/// takes no cut of every row: it reads the rows it may tell. One that
/// states neither would copy every row anyway, so it takes the feed's
/// cut, which a call let in while it runs shares. Beside another call
/// each takes the cut the two share, once: the first of them takes it,
/// whatever it states. So does a call beside an open watch, where no cut
/// stands at the log's head: the watch can share it.
#[tokio::test]
async fn a_lone_snapshot_takes_a_cut_only_where_it_tells_every_row() {
    use proto::snapshot_request::Slice;

    let (_elector, _standing, control) = electing(true);
    let feed = control.shared.feed.clone();
    let models = ["model-a", "model-a", "model-a", "model-b"];
    for (worker, model) in models.into_iter().enumerate() {
        let request = proto::RegisterWorkerRequest {
            spec: Some(proto::WorkerSpec {
                url: format!("grpc://10.0.0.{worker}:8000"),
                wire: proto::WireMode::Grpc as i32,
                role: proto::Role::Single as i32,
                models: vec![model.to_owned()],
                ..Default::default()
            }),
            ..Default::default()
        };
        control
            .register_worker(from("operator-a", request))
            .await
            .expect("registered");
    }
    let told = async |limit| {
        let mut request = snapshot(&[Slice::Workers]);
        request.get_mut().limit = limit;
        let answer = control
            .get_snapshot(request)
            .await
            .expect("the rows")
            .into_inner();
        (answer.workers.len(), answer.truncated)
    };
    assert_eq!(told(1).await, (1, true));
    assert_eq!(feed.cuts(), 0, "a lone call for one row copied every row");
    assert_eq!(told(0).await, (4, false));
    assert_eq!(feed.cuts(), 1, "a call for every row takes the cut");
    assert!(feed.kept().is_none(), "no reader can share the cut");

    // A lone call for one model with no limit reads that model's rows
    // alone, and takes no cut.
    let mut request = snapshot(&[Slice::Workers]);
    request.get_mut().models = vec!["model-a".to_owned()];
    let answer = control
        .get_snapshot(request)
        .await
        .expect("the rows")
        .into_inner();
    assert_eq!((answer.workers.len(), answer.truncated), (3, false));
    assert_eq!(feed.cuts(), 1, "a lone call for its models took the cut");

    // Another call is let in beside it: the call for one row takes the
    // cut the two share, and the call behind it shares that one.
    let beside = control
        .shared
        .readers
        .snapshots()
        .place()
        .expect("a place is free");
    assert_eq!(told(1).await, (1, true));
    assert_eq!(feed.cuts(), 2, "a call beside another takes their cut");
    assert_eq!(told(0).await, (4, false));
    assert_eq!(feed.cuts(), 2, "one copy for the calls let in");
    drop(beside);

    // A watch is open, and no cut stands at the log's head: the call
    // for one row takes the cut the watch can share, and keeps it.
    assert_eq!(told(0).await, (4, false));
    assert!(feed.kept().is_none(), "the last call out let the cut go");
    let gateways = Service::new(vec![Scope::Gateway], Arc::clone(&control.shared));
    let watching = gateways
        .watch_workers(from("gateway-a", proto::WatchWorkersRequest::default()))
        .await
        .expect("opened");
    assert_eq!(told(1).await, (1, true));
    assert_eq!(feed.cuts(), 3, "a call beside an open watch takes its cut");
    assert!(feed.kept().is_some(), "kept for the watch");
    drop(watching);
}

/// A snapshot that waited its turn through a later take of the lease is
/// told to retry: the lead is read again behind the wait, and no answer
/// is told under a term the store has since passed.
#[tokio::test]
async fn a_snapshot_that_waited_through_a_later_take_is_told_to_retry() {
    use proto::snapshot_request::Slice;

    let (_elector, _standing, control) = electing(true);
    let control = Arc::new(control);
    let held = control.shared.readers.snapshots().hold().await;
    let mut calls = JoinSet::new();
    calls.spawn({
        let control = Arc::clone(&control);
        async move { control.get_snapshot(snapshot(&[Slice::Workers])).await }
    });
    // The call is let in and waits behind the held turn.
    let began = Instant::now();
    while control.shared.readers.snapshots().let_in() < 1 {
        assert!(
            began.elapsed() < Duration::from_secs(20),
            "the call took no place"
        );
        tokio::task::yield_now().await;
    }
    // The store is told a later take than the call was admitted under.
    control.shared.store.set_term(2);
    drop(held);
    let refused = calls
        .join_next()
        .await
        .expect("the one call")
        .expect("the call's task")
        .expect_err("a later take");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::Unavailable, super::super::TAKE_OVER)
    );
}

/// A snapshot admitted under one term waits its turn while the replica
/// gives its lease up and takes it again, its store told the new term.
/// The fence and the store both stand in the later term: the lead read
/// again names a term, and it is not the call's, so the compare itself
/// tells the call to retry.
#[tokio::test]
async fn a_snapshot_that_waited_through_a_take_its_store_follows_is_told_to_retry() {
    use proto::snapshot_request::Slice;

    let (mut elector, _standing, control) = electing(true);
    let control = Arc::new(control);
    let held = control.shared.readers.snapshots().hold().await;
    let mut calls = JoinSet::new();
    calls.spawn({
        let control = Arc::clone(&control);
        async move { control.get_snapshot(snapshot(&[Slice::Workers])).await }
    });
    let began = Instant::now();
    while control.shared.readers.snapshots().let_in() < 1 {
        assert!(
            began.elapsed() < Duration::from_secs(20),
            "the call took no place"
        );
        tokio::task::yield_now().await;
    }
    let now = Instant::now();
    assert!(elector.step_down(None, now).expect("the release"));
    assert_eq!(
        elector.tick(now).expect("the take"),
        TickOutcome::Acquired { term: 2 }
    );
    control.shared.store.set_term(2);
    assert_eq!(control.watched(Instant::now()).ok(), Some(2));
    drop(held);
    let refused = calls
        .join_next()
        .await
        .expect("the one call")
        .expect("the call's task")
        .expect_err("a later take");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::Unavailable, super::super::TAKE_OVER)
    );
}

/// The snapshot's header names the term its call was admitted under,
/// whichever term that is. A replica that gave its lease up and took it
/// again leads its second term, its store told it: the header that named
/// one names two.
#[tokio::test]
async fn a_snapshot_under_a_second_take_names_the_second_term() {
    use proto::snapshot_request::Slice;

    let (mut elector, _standing, control) = electing(true);
    let first = control
        .get_snapshot(snapshot(&[Slice::Workers]))
        .await
        .expect("a snapshot")
        .into_inner();
    assert_eq!(first.term, 1);
    let now = Instant::now();
    assert!(elector.step_down(None, now).expect("the release"));
    assert_eq!(
        elector.tick(now).expect("the take"),
        TickOutcome::Acquired { term: 2 }
    );
    control.shared.store.set_term(2);
    let second = control
        .get_snapshot(snapshot(&[Slice::Workers]))
        .await
        .expect("a snapshot")
        .into_inner();
    assert_eq!(second.term, 2);
}

/// A row's age in the snapshot is its age at the call. A worker is
/// registered, a moment passes and the snapshot is asked for: the age it
/// tells lies between the two spans the case measures around it.
#[tokio::test]
async fn a_snapshot_tells_a_rows_age_at_the_call() {
    use proto::snapshot_request::Slice;

    let (_elector, _standing, control) = electing(true);
    let spec = proto::WorkerSpec {
        url: "grpc://10.0.0.5:8000".to_owned(),
        wire: proto::WireMode::Grpc as i32,
        role: proto::Role::Single as i32,
        models: vec!["model-a".to_owned()],
        ..Default::default()
    };
    let request = proto::RegisterWorkerRequest {
        spec: Some(spec),
        ..Default::default()
    };
    let before = Instant::now();
    control
        .register_worker(from("operator-a", request))
        .await
        .expect("registered");
    let admitted = Instant::now();
    tokio::time::sleep(Duration::from_millis(20)).await;
    let asked = Instant::now();
    let answer = control
        .get_snapshot(snapshot(&[Slice::Workers]))
        .await
        .expect("a snapshot")
        .into_inner();
    let told = Instant::now();
    let [row] = answer.workers.as_slice() else {
        panic!("one row: {answer:?}");
    };
    // The row was admitted between `before` and `admitted`, and the
    // snapshot read its clock between `asked` and `told`.
    let age = u128::from(row.admitted_age_ms);
    let least = asked.duration_since(admitted).as_millis();
    let most = told.duration_since(before).as_millis();
    assert!(
        (least..=most).contains(&age),
        "an age of {age} ms, outside {least} to {most} ms"
    );
}
