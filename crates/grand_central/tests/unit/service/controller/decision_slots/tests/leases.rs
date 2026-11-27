//! The cases of the lease a decision files and of the leases it reads: what
//! an answer carries of its lease, what the leases on file book and when a
//! decision reads them, the answer that files none, and the abandoned
//! lease, which is told to no stage that a later generation built new.

use super::*;

/// A decision waits in its stream's queue as it was made, with the
/// lease issued for its pick and what is left of that lease's span: its
/// frame is its stream's to build. A `Busy` is handed over as its
/// frame.
#[tokio::test]
async fn a_decision_is_handed_to_its_stream_as_it_was_made_and_a_busy_as_its_frame() {
    let steps = [Duration::ZERO, Duration::ZERO, BUDGET].map(Step::At);
    let mut bed = leading(steps);
    assert_eq!(bed.submit(frame(1, 1)), None);
    let handed = timeout(PATIENCE, bed.answers.recv()).await;
    let Ok(Some(wire::Decided::Decision {
        id,
        term,
        decision,
        lease: Some(leased),
        encode: None,
    })) = handed
    else {
        panic!("a decision as it was made, with its lease: {handed:?}");
    };
    assert_eq!((id.as_slice(), term), (&[1][..], 1));
    let first = decision.pick().and_then(|pick| pick.targets.first());
    let first = first.expect("a pick names a target");
    assert!(bed.workers.contains(&first.id), "{decision:?}");
    assert_eq!(leased.lease.id, LeaseId(1 << 32 | 1));
    assert_eq!(leased.left, DISPATCH_TTL);
    // A request taken with its budget gone is answered where it is
    // taken, and the answer carries no list.
    assert_eq!(bed.submit(frame(2, 2)), None);
    let handed = timeout(PATIENCE, bed.answers.recv()).await;
    let Ok(Some(wire::Decided::Frame(told))) = handed else {
        panic!("a Busy as its frame: {handed:?}");
    };
    assert_eq!(busy(&told), Reason::BudgetGoneOnArrival);
    assert_eq!(told.request_id, [2]);
}

/// A member's requests in flight are the leases on file that book it:
/// a decision by the fewest requests goes where the decisions before
/// it left the fewest, and a lease that ends gives its worker the next.
#[tokio::test]
async fn a_decision_by_the_fewest_requests_reads_the_leases_the_decisions_before_it_left() {
    let mut document = PolicySpec::power_of_two(PowerOfTwoMetric::LeastRequests)
        .expect("a form of this build's preset");
    document.generation = 1;
    let mut bed = bed(Some(applied(&document)), true, (2, 8), []);
    // Nine decisions and no word from their gateway: each round of
    // three names each of the three members once.
    let mut held = Vec::new();
    for round in 0..3u8 {
        let mut named = BTreeSet::new();
        for turn in 1..=3u8 {
            let id = round * 3 + turn;
            let answer = bed.ask(frame(id, u64::from(id))).await;
            named.insert(picked(&answer));
            held.push((picked(&answer), LeaseId(lease_of(&answer).id)));
        }
        assert_eq!(named.len(), 3, "round {round} named {named:?}");
    }
    // One member's three leases end at their gateway's word: the next
    // three decisions are that member's.
    let (freed, _) = held[0];
    for &(_, lease) in held.iter().filter(|(worker, _)| *worker == freed) {
        let ended = bed
            .store
            .released(GATEWAY, lease, None, ReleaseReason::ClientGone);
        assert!(ended, "{lease:?} was on file");
    }
    for id in 10..=12u8 {
        let answer = bed.ask(frame(id, u64::from(id))).await;
        assert_eq!(picked(&answer), freed, "request {id}");
    }
}

/// A decision reads its rows behind what was handed while its decider
/// worked: a release handed while the decider is held at a request is
/// applied by that request ahead of its reads. Under the fewest
/// requests the request then picks as the first one did under its
/// seed: that worker stands at no request again.
#[tokio::test]
async fn a_decision_reads_its_rows_behind_what_was_handed_ahead_of_it() {
    let mut document = PolicySpec::power_of_two(PowerOfTwoMetric::LeastRequests)
        .expect("a form of this build's preset");
    document.generation = 1;
    let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
    let first = bed.ask(frame(1, 7)).await;
    // The second request is held at its read of the clock: its decider
    // is at work, and what is handed is left to it.
    let (reached, held) = std_mpsc::sync_channel(1);
    let (go, word) = std_mpsc::sync_channel(1);
    let hold = Step::Hold { reached, go: word };
    bed.script.steps.lock().push_back(hold);
    assert_eq!(bed.submit(frame(2, 7)), None);
    held.recv_timeout(PATIENCE)
        .expect("the decider reads the clock");
    let gone = Said::Released(None, ReleaseReason::ClientGone);
    let hand = Hand::new(GATEWAY);
    bed.store.hand(&hand, LeaseId(lease_of(&first).id), gone);
    let live = LeasesStore::live(&*bed.store, GATEWAY);
    assert_eq!(live, 1, "the hand leaves the release to the decider");
    go.send(()).expect("the decider holds the other end");
    let second = bed.answer().await;
    assert_eq!(picked(&second), picked(&first), "no request stands there");
}

#[tokio::test]
async fn a_picks_answer_carries_the_lease_the_store_filed_under_the_fences_term() {
    let mut bed = leading([]);
    let answer = bed.ask(frame(7, 7)).await;
    let lease = *lease_of(&answer);
    assert_eq!(lease.id, 1 << 32 | 1, "the term over its first count");
    assert_eq!(
        (lease.term, lease.origin(), lease.kind(), lease.side()),
        (
            answer.term,
            proto::LeaseOrigin::Controller,
            proto::LeaseKind::Credit,
            proto::Side::Single
        )
    );
    assert_eq!(
        (lease.target_index, lease.generation, lease.session_key),
        (0, 1, 0)
    );
    assert_eq!(u128::from(lease.dispatch_ttl_ms), DISPATCH_TTL.as_millis());
    let store: &dyn LeasesStore = bed.store.as_ref();
    let filed = store
        .get(GATEWAY, LeaseId(lease.id))
        .expect("the lease is on file");
    assert_eq!(
        (filed.target, filed.term, filed.state, filed.request),
        (picked(&answer), 1, LeaseState::ISSUED, fingerprint(&[7]))
    );
    assert_eq!(filed.dispatch_deadline, filed.issued_at + DISPATCH_TTL);
    // The next decision's id is the term's next count, its lease
    // carries its request's prompt tokens, and both leases stand for
    // their gateway alone.
    assert_eq!(lease.tokens, 0, "a request that states no prompt");
    let mut prompted = frame(8, 8);
    let facts = prompted.facts.as_mut();
    let facts = facts.expect("the case's frame has facts");
    facts.prompt_tokens = 48;
    facts.session_key = Some(proto::Identity {
        form: Some(proto::identity::Form::Hash(17)),
    });
    let next = bed.ask(prompted).await;
    assert_eq!(
        (lease_of(&next).id, lease_of(&next).tokens),
        (1 << 32 | 2, 48)
    );
    let store: &dyn LeasesStore = bed.store.as_ref();
    assert_eq!((store.live(GATEWAY), store.live(OTHER)), (2, 0));
    // The lease carries its request's session, on the wire as on file:
    // never zero for a request that states one.
    let session = lease_of(&next).session_key;
    let filed = store.get(GATEWAY, LeaseId(1 << 32 | 2));
    let filed = filed.expect("the lease is on file").lease;
    assert_ne!(session, 0);
    assert_eq!(filed.session.get(), session);
}

/// Requests that state no prompt size spread over the members: each
/// dispatch is its worker's credit at the load stage's mean until that
/// worker's next report, and each lease states no size.
#[tokio::test]
async fn a_dispatched_request_that_stated_no_size_is_priced_until_its_workers_next_report() {
    let mut bed = bed(Some(preset("least_load", 1)), true, (2, 8), []);
    // Each member's report, sampled ahead of every send, at one rate:
    // the first holds no work and the other two a quarter of a mean
    // prompt each. The load monitor publishes the three.
    let at = bed.script.taken;
    let loads = ["w0", "w1", "w2"].iter().enumerate();
    let loads = loads.map(|(turn, name)| {
        let mut report = one_rank_load(if turn == 0 { 0 } else { 256 }, 0.1, 1_000.0);
        report.sampled_at = Some(at - Duration::from_millis(1));
        (format!("http://{name}:8000"), report)
    });
    bed.loads
        .send_replace(LoadSnapshot::from_loads_for_test(loads.collect()));
    // Three requests that state no prompt size, each dispatched before
    // the next is decided. A dispatch is its worker's credit until that
    // worker's next report, and the load stage credits one of no stated
    // size at its mean: at no tokens it would cost nothing there, and
    // each request would read the first member as free.
    let mut named = Vec::new();
    for id in 1..=3u8 {
        let answer = bed.ask(frame(id, u64::from(id))).await;
        assert_eq!(
            lease_of(&answer).tokens,
            0,
            "the lease states what the request did"
        );
        named.push(picked(&answer));
        let frame = proto::Dispatched {
            lease_id: lease_of(&answer).id,
            target_index: 0,
            side: proto::Side::Single.into(),
            since_decision_micros: 250,
        };
        let hand = Hand::new(GATEWAY);
        lifecycle::dispatched(&bed.store, &hand, &frame, bed.script.taken)
            .expect("the frame reads");
    }
    assert_eq!(named[0], bed.workers[0], "the member with no work");
    let members: BTreeSet<WorkerId> = named.iter().copied().collect();
    assert_eq!(members.len(), 3, "each member took one request: {named:?}");
}

#[tokio::test]
async fn an_answer_that_names_no_worker_or_does_not_leave_files_no_lease() {
    // A shed: no member serves the model.
    let mut bed = leading([]);
    let answer = bed.ask(frame_for("model-none", 1, 1)).await;
    assert!(matches!(outcome(&answer), Outcome::Shed(_)), "{answer:?}");
    assert_eq!(unleased(answer.clone()), answer, "a shed carries no lease");
    assert_eq!(bed.store.live(GATEWAY), 0);
    // A decision whose fence closed at its hand-over, and one whose
    // budget ended while it was made: `Busy`, and nothing on file.
    for (second_read, reason) in [
        (FENCED, Reason::NotLeader),
        (BUDGET, Reason::BudgetGoneOnArrival),
    ] {
        let mut bed = leading([Step::At(Duration::ZERO), Step::At(second_read)]);
        assert_eq!(busy(&bed.ask(frame(2, 2)).await), reason);
        assert_eq!(bed.store.live(GATEWAY), 0, "{reason:?}");
    }
}

#[tokio::test]
async fn a_pick_the_store_mints_no_lease_for_leaves_as_busy() {
    // The term's last id, then none: `Busy` with no reason, counted.
    let mut bed = leading([]);
    bed.store.spend_ids(1);
    let last = bed.ask(frame(1, 1)).await;
    assert_eq!(lease_of(&last).id, 1 << 32 | u64::from(u32::MAX));
    for id in 2..=3 {
        assert_eq!(busy(&bed.ask(frame(id, 1)).await), Reason::Unspecified);
    }
    assert_eq!(bed.counts.unleased.load(Ordering::Relaxed), 2);
    assert_eq!(bed.store.live(GATEWAY), 1, "the last id's lease alone");
    // A store that stands under a later term than the fence showed:
    // not this leader's to answer, and no lease.
    let mut bed = leading([]);
    bed.store.set_term(2);
    assert_eq!(busy(&bed.ask(frame(4, 1)).await), Reason::NotLeader);
    assert_eq!(bed.counts.unleased.load(Ordering::Relaxed), 0);
    assert_eq!(bed.store.live(GATEWAY), 0);
}

/// An abandoned lease is told to the engine whose generation issued it
/// and to no stage a later one built new: such a stage never saw the
/// issue. Each generation here is validated over a stage table of its
/// own, so the later one keeps none. The lease is ended all the same.
#[tokio::test]
async fn a_later_generations_engine_is_not_told_of_a_lease_it_did_not_issue() {
    let told = Arc::new(Mutex::new(Vec::new()));
    let spec = |generation: u64| hearing(&told, generation);
    let mut bed = bed(Some(spec(1)), true, (2, 8), []);
    let id = |count: u64| 1 << 32 | count;
    bed.ask(frame(1, 1)).await;
    // The next generation stands when the first lease's deadline is
    // past: its engine is told of its own issue and of nothing else.
    bed.store.apply(spec(2)).expect("a later generation");
    let late = DISPATCH_TTL + Duration::from_secs(1);
    bed.script.steps.lock().push_back(Step::At(late));
    assert_eq!(bed.submit_at(frame(2, 1), GATEWAY, late), None);
    bed.answer().await;
    assert_eq!(*told.lock(), [(id(1), None), (id(2), None)]);
    let store: &dyn LeasesStore = bed.store.as_ref();
    assert_eq!(store.get(GATEWAY, LeaseId(id(1))), None);
    assert_eq!(store.live(GATEWAY), 1);
}

/// A decision reads what the leases on file book, whatever its profile
/// and with nothing declared: the three leases the decisions before it
/// left are counted on their workers' objects, and its records carry
/// each worker's count.
#[tokio::test]
async fn a_decision_reads_the_leases_on_file_in_its_workers_records() {
    let rows = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&rows);
    let mut catalog = StageRegistry::builtin();
    let seen = move || -> Box<dyn Picker> {
        let rows = Arc::clone(&kept);
        Box::new(Seen { rows })
    };
    catalog.picker("seen", seen).expect("a free name");
    let mut document = PolicySpec::preset("random").expect("a preset of this build");
    let profile = document.profiles.get_mut("random");
    profile.expect("the preset's profile").pick = "seen".into();
    let digest = document.digest().expect("the document digests");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[])
        .expect("the catalog holds the stage");
    let mut bed = bed(Some(spec), true, (2, 8), []);
    // Three decisions and no word from their gateway: three leases
    // stand on file when the fourth is made.
    for request in 1..=4u8 {
        bed.ask(frame(request, u64::from(request))).await;
    }
    // What the fourth decision was handed: its facts read the three
    // leases, in its records.
    let rows = rows.lock();
    let (read, own): (Vec<u32>, Vec<u32>) = rows[3].iter().copied().unzip();
    assert_eq!(read.iter().sum::<u32>(), 3, "{rows:?}");
    assert_eq!(own.iter().sum::<u32>(), 3, "{rows:?}");
}

/// A decision can pick a worker the store's world does not list yet:
/// the worker objects follow a registration as its write lands, and
/// the store builds the pools that list it a window later. The
/// lease of such a pick books the worker's count all the same, by the
/// slot the registration filed: the next decision reads one request in
/// flight on the worker, on its object and in its record. A lease that
/// booked nothing would leave every decision of the window reading the
/// new worker idle.
#[tokio::test]
async fn a_pick_inside_its_workers_registration_window_books_its_count() {
    const LATE: &str = "model-late";
    let rows = Arc::new(Mutex::new(Vec::new()));
    let kept = Arc::clone(&rows);
    let spec = picking("seen", move || -> Box<dyn Picker> {
        let rows = Arc::clone(&kept);
        Box::new(Seen { rows })
    });
    let mut bed = bed(Some(spec), true, (1, 8), []);
    // One more member, alone under its model, written ready and never
    // published: the build that lists it is still owed.
    let key = String::from("http://late:8000");
    let facts = WorkerFacts {
        key: key.clone(),
        engine: key,
        incarnation: 1,
        role: Role::Single,
        models: vec![LATE.to_owned()],
        dp_size: 1,
        max_running: Some(4),
        ..WorkerFacts::default()
    };
    let late = rewritten(&bed, facts);
    let row = WorkersStore::get(&*bed.store, late).expect("the row");
    bed.reads
        .objects
        .connect_row(&bed.store, late, row.revision);
    let listed = bed.store.snapshot().booked_of(late);
    assert_eq!(listed, None, "no world lists the worker yet");
    for request in 1..=2u8 {
        let answer = bed.ask(frame_for(LATE, request, u64::from(request))).await;
        assert_eq!(picked(&answer), late);
    }
    // What the second decision was handed: the first one's lease, in
    // the count its facts read and in the worker's record.
    assert_eq!(rows.lock()[1], [(1, 1)]);
}
