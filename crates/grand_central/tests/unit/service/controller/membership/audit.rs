//! The cases of the audit lines: the line every call leaves, written,
//! unmoved or refused, a line a row for a registration of several ranks, a
//! removal that health asked, a refresh in its own words, a write refused
//! where the replica does not lead, a refusal's line behind the turn, and
//! the line every write of a provider leaves.

use super::*;

#[test]
fn every_call_leaves_its_audit_line_written_unmoved_or_refused() {
    // A subscriber of the case's own keeps this thread's lines. A
    // second one stays registered beside it: with one alone a line is
    // on or off by the thread that reaches it first, and a case beside
    // this one reaches it under none.
    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(subscriber);

    let (_store, door) = door();
    let add = ask(spec(URL), Mode::Add);
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &add)
            .expect("registered"),
    );
    door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Upsert))
        .expect("unmoved");
    door.register(&operator(), under(TERM), &add)
        .expect_err("the URL stands");
    let nothing = proto::PatchWorkerRequest {
        worker_id: row.id,
        revision: row.revision,
        ..Default::default()
    };
    door.patch(&operator(), under(TERM), &nothing)
        .expect("unmoved");
    let drain = proto::DeregisterWorkerRequest {
        worker_id: row.id,
        revision: row.revision,
        drain: true,
    };
    for answer in ["draining", "draining still"] {
        door.deregister(&operator(), under(TERM), &drain)
            .expect(answer);
    }
    // A line a call, in the calls' order: what a write made, `nothing`
    // where a call wrote none, a refusal by its code.
    let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
    let lines: Vec<&str> = logged.lines().collect();
    let want = [
        ("membership write", "RegisterWorker", "made=\"registered\""),
        ("membership write", "RegisterWorker", "made=\"nothing\""),
        (
            "membership call refused",
            "RegisterWorker",
            "code=AlreadyExists",
        ),
        ("membership write", "PatchWorker", "made=\"nothing\""),
        ("membership write", "DeregisterWorker", "made=\"status\""),
        ("membership write", "DeregisterWorker", "made=\"nothing\""),
    ];
    assert_eq!(lines.len(), want.len(), "{logged}");
    for (line, (kind, call, made)) in lines.iter().zip(want) {
        let call = format!("call=\"{call}\" by=operator-a");
        assert!(
            line.contains(kind) && line.contains(&call) && line.contains(made),
            "{line}"
        );
    }
    // No line holds the URL: it may carry a credential.
    assert!(!logged.contains(URL), "{logged}");
}

/// A registration that states several ranks leaves a line a row: what
/// the call made of each rank, `nothing` where it moved none, and each
/// row that left.
#[test]
fn a_registration_of_several_ranks_leaves_an_audit_line_a_row() {
    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(subscriber);

    let (_store, door) = door();
    for (width, mode) in [(3, Mode::Add), (3, Mode::Upsert), (2, Mode::Upsert)] {
        door.register(&operator(), under(TERM), &ask(wide(URL, width), mode))
            .expect("laid");
    }
    let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
    let made: Vec<&str> = logged
        .lines()
        .map(|line| {
            let call = "call=\"RegisterWorker\" by=operator-a";
            assert!(
                line.contains("membership write") && line.contains(call),
                "{line}"
            );
            ["registered", "relabelled", "nothing", "removed"]
                .into_iter()
                .find(|made| line.contains(&format!("made=\"{made}\"")))
                .expect("what the call made of the row")
        })
        .collect();
    assert_eq!(
        made,
        [
            "registered",
            "registered",
            "registered",
            "nothing",
            "nothing",
            "nothing",
            "relabelled",
            "relabelled",
            "removed"
        ]
    );
    // No line holds the URL: it may carry a credential.
    assert!(!logged.contains(URL), "{logged}");
}

#[test]
fn a_removal_health_asked_leaves_an_audit_line_that_names_health() {
    use tracing::{
        subscriber::{self, NoSubscriber},
        Dispatch,
    };

    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let _beside = Dispatch::new(NoSubscriber::default());
    let _kept = subscriber::set_default(subscriber);

    let (store, door) = hosted_door();
    let key = "private/health-key";
    let maker = found(key);
    door.found_one(&maker, URL, &|_| found_facts())
        .expect("registered");
    let row = store.by_key(URL).expect("the row health saw");
    let seen = RowRevision {
        row: row.id,
        revision: row.revision,
    };
    assert_eq!(
        door.drained(Remover::Health(UnhealthyRemovalScope::Disabled), URL, seen),
        Err(MembershipWriteRefused::OutOfScope)
    );
    let by = Remover::Health(UnhealthyRemovalScope::DiscoveryOwned);
    assert_eq!(
        door.drained(by, URL, seen),
        Ok(MembershipWriteOutcome::Draining(row.id))
    );
    assert_eq!(
        door.gone(by, URL, seen),
        Ok(MembershipWriteOutcome::Removed(row.id))
    );

    let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
    let calls: Vec<_> = logged.lines().skip(1).collect();
    assert_eq!(calls.len(), 3, "{logged}");
    assert!(
        calls.iter().all(|line| line.contains("asker=\"health\"")),
        "{logged}"
    );
    assert!(calls[0].contains("code=PermissionDenied"), "{logged}");
    assert!(calls[1].contains("made=\"status\""), "{logged}");
    assert!(calls[2].contains("made=\"removed\""), "{logged}");
    assert!(!logged.contains(URL) && !logged.contains(key), "{logged}");
}

/// A refresh leaves one audit line a call, under its own name. One that
/// states another fact than the labels is refused in a refresh's
/// words, never in a removal's, and no line holds the URL.
#[test]
fn a_refresh_leaves_its_audit_line_in_its_own_words() {
    const HTTP: &str = "http://10.0.0.5:8000";
    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(subscriber);

    let (store, door) = hosted_door();
    let built = |model: &str, tier: &str| -> Arc<dyn Worker> {
        let worker = <BasicWorkerBuilder>::new(HTTP).model(ModelCard::new(model));
        Arc::new(worker.label("tier", tier).build())
    };
    let first = built("m", "gold");
    door.found_built(&Registrant::Control, Precondition::Absent, &first, None)
        .expect("registered");
    let row = store.by_key(HTTP).expect("the registered row");
    let seen = standing(&row);
    let stray = door.refreshed(seen, &built("other", "gold"), &first);
    assert_eq!(stray, Err(MembershipWriteRefused::NotLabelsAlone));
    let wrote = door.refreshed(seen, &built("m", "silver"), &first);
    assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));

    let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
    let calls: Vec<_> = logged.lines().skip(1).collect();
    assert_eq!(calls.len(), 2, "{logged}");
    assert!(
        calls
            .iter()
            .all(|line| line.contains("worker label refresh")),
        "{logged}"
    );
    assert!(calls[0].contains("code=PermissionDenied"), "{logged}");
    assert!(
        calls[0].contains(MembershipWriteRefused::NOT_LABELS_ALONE),
        "{logged}"
    );
    assert!(calls[1].contains("made=\"relabelled\""), "{logged}");
    assert!(!logged.contains(HTTP), "{logged}");
}

#[test]
fn every_write_of_a_provider_leaves_its_audit_line() {
    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(subscriber);

    let (store, door) = door();
    let hosted = ReplicaMembershipWriter::new(Arc::new(door), operator(), under(TERM));
    let key = "private/provider-key";
    let maker = found(key);
    let rank_urls = ["grpc://10.0.0.5:8000@0", "grpc://10.0.0.5:8000@1"];
    let states = [0, 1].map(|rank: u16| {
        let key = rank_urls[usize::from(rank)];
        move |_: Option<&WorkerFacts>| WorkerFacts {
            key: key.to_owned(),
            dp_rank: rank,
            dp_size: 2,
            ..found_facts()
        }
    });
    let ranks = [0, 1].map(|rank| Stated {
        url: rank_urls[rank],
        state: &states[rank],
    });
    hosted
        .rows(&maker, Precondition::Any, &ranks)
        .expect("registered ranks");
    let [first, second] = rank_urls.map(|url| store.by_key(url).expect("a rank").id.0);
    hosted
        .rows(&maker, Precondition::Any, &ranks)
        .expect("unmoved ranks");

    hosted
        .rows(
            &maker,
            Precondition::Any,
            &[Stated {
                url: URL,
                state: &|_| found_facts(),
            }],
        )
        .expect("registered plain and retired ranks");
    let id = store.by_key(URL).expect("the registered row").id.0;
    let refreshed = || WorkerFacts {
        max_running: Some(8),
        ..found_facts()
    };
    hosted
        .found_one(&maker, URL, &|_| refreshed())
        .expect("relabelled");
    hosted
        .found_one(&maker, URL, &|_| WorkerFacts {
            role: Role::Decode,
            ..refreshed()
        })
        .expect("replaced");
    let row = store.by_key(URL).expect("the replaced row");
    let new = row.id.0;
    let seen = standing(&row);
    let by = Remover::Registrant(&maker);
    hosted.drained(by, URL, seen).expect("drained");
    hosted.drained(by, URL, seen).expect("unmoved");
    hosted
        .gone(Remover::Registrant(&found("another/key")), URL, seen)
        .expect_err("another maker");
    hosted
        .gone(Remover::Provider("cluster"), URL, seen)
        .expect("gone");
    hosted.gone(by, URL, seen).expect_err("already gone");

    let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
    let lines: Vec<&str> = logged.lines().collect();
    let registration = "worker registration";
    let want = [
        (registration, "registered", Some((first, 1, 2))),
        (registration, "registered", Some((second, 2, 3))),
        (registration, "nothing", Some((first, 2, 3))),
        (registration, "nothing", Some((second, 2, 3))),
        (registration, "registered", Some((id, 3, 4))),
        (registration, "removed", Some((first, 4, 5))),
        (registration, "removed", Some((second, 5, 6))),
        (registration, "relabelled", Some((id, 6, 7))),
        (registration, "replaced", Some((new, 7, 8))),
        ("worker drain", "status", Some((new, 8, 9))),
        ("worker drain", "nothing", Some((new, 8, 9))),
        (
            "worker removal",
            "code=PermissionDenied why=\"the row is provider cluster's under another key",
            None,
        ),
        ("worker removal", "removed", Some((new, 9, 10))),
        (
            "worker removal",
            "code=NotFound why=\"no worker is registered under the URL\"",
            None,
        ),
    ];
    assert_eq!(lines.len(), want.len(), "{logged}");
    for (line, (call, detail, place)) in lines.iter().zip(want) {
        assert!(
            line.contains(&format!("call=\"{call}\" by=operator-a")),
            "{line}"
        );
        assert_eq!(
            line.contains("asker=\"cluster\""),
            call != registration,
            "{line}"
        );
        if let Some((worker, epoch, seq)) = place {
            let fields = format!("worker={worker:016x} made=\"{detail}\" membership_epoch={epoch} seq={seq} term={TERM}");
            assert!(
                line.contains("membership write") && line.contains(&fields),
                "{line}"
            );
        } else {
            assert!(
                line.contains("membership call refused") && line.contains(detail),
                "{line}"
            );
        }
    }
    for private in [URL, key, "another/key"] {
        assert!(!logged.contains(private), "{logged}");
    }
}

/// A provider's write on a replica that does not lead leaves a refusal's
/// audit line under the call's own name: the code and the words that
/// call's read of the fence answered, and an asker on the drain's line and
/// on the removal's alone. No line holds the URL or the provider's key.
#[test]
fn a_write_refused_where_the_replica_does_not_lead_leaves_its_audit_line() {
    // What the fence answers at each read, another code or other words
    // each time: a line carries the code and the words of its own call.
    const READS: [(Code, &str); 3] = [
        (Code::FailedPrecondition, "this replica does not lead"),
        (Code::Unavailable, "the lease of this replica ran out"),
        (Code::FailedPrecondition, "this replica follows another"),
    ];
    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(subscriber);

    let (_store, door) = door();
    let reads = AtomicUsize::new(0);
    let follower = ReplicaMembershipWriter::new(Arc::new(door), operator(), move || {
        let (code, words) = READS[reads.fetch_add(1, Ordering::Relaxed)];
        Err(Status::new(code, words))
    });
    let key = "private/provider-key";
    let maker = found(key);
    let refused = |read: usize| {
        let (_, words) = READS[read];
        Err(MembershipWriteRefused::NotLeading(words.to_owned()))
    };
    assert_eq!(
        follower.found_one(&maker, URL, &|_| found_facts()),
        refused(0)
    );
    let seen = RowRevision {
        row: WorkerId(1),
        revision: 1,
    };
    let by = Remover::Provider("cluster");
    assert_eq!(follower.drained(by, URL, seen), refused(1));
    assert_eq!(follower.gone(by, URL, seen), refused(2));

    let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
    let lines: Vec<&str> = logged.lines().collect();
    let calls = [
        ("worker registration", false),
        ("worker drain", true),
        ("worker removal", true),
    ];
    assert_eq!(lines.len(), calls.len(), "{logged}");
    for ((line, (call, asked)), (code, words)) in lines.iter().zip(calls).zip(READS) {
        let refusal = format!("call=\"{call}\" by=operator-a code={code:?} why=\"{words}\"");
        assert!(
            line.contains("membership call refused") && line.contains(&refusal),
            "{line}"
        );
        assert_eq!(line.contains("asker="), asked, "{line}");
        assert!(!asked || line.contains("asker=\"cluster\""), "{line}");
    }
    for private in [URL, key] {
        assert!(!logged.contains(private), "{logged}");
    }
}

/// A refusal's audit line is written once its call has let the door's
/// turn go, at both doors: the line of a control call the fence refuses,
/// of a provider's write the fence refuses, and of a provider's write the
/// door refuses under a term that leads.
#[test]
fn a_refusals_audit_line_is_written_once_the_turn_is_let_go() {
    let (_store, door) = door();
    let door = Arc::new(door);
    let maker = found("pods/a");
    // The row stands before the lines are kept: a write's own line is
    // written inside the turn.
    let leader = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM));
    leader
        .found_one(&maker, URL, &|_| found_facts())
        .expect("registered");

    // The lines' writer reads the door's turn as each line reaches it.
    let lines = Logged::default();
    let held = Arc::new(AtomicUsize::new(0));
    let (writer, watched, seen) = (lines.clone(), Arc::clone(&door), Arc::clone(&held));
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || {
            seen.fetch_add(usize::from(watched.turn.is_locked()), Ordering::Relaxed);
            writer.clone()
        })
        .finish();
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(subscriber);

    let follows = || -> Result<u64, Status> {
        Err(Status::failed_precondition("this replica does not lead"))
    };
    door.register(&operator(), follows, &ask(spec(URL), Mode::Add))
        .expect_err("the fence refuses the control call");
    ReplicaMembershipWriter::new(Arc::clone(&door), operator(), follows)
        .found_one(&maker, URL, &|_| found_facts())
        .expect_err("the fence refuses the provider's write");
    assert_eq!(
        leader.found(&maker, Precondition::Absent, URL, &|_| found_facts()),
        Err(MembershipWriteRefused::Stands)
    );

    // A line a refusal, in the calls' order, each told by its call and
    // its code.
    let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
    let lines: Vec<&str> = logged.lines().collect();
    let want = [
        ("RegisterWorker", "FailedPrecondition"),
        ("worker registration", "FailedPrecondition"),
        ("worker registration", "AlreadyExists"),
    ];
    assert_eq!(lines.len(), want.len(), "{logged}");
    for (line, (call, code)) in lines.iter().zip(want) {
        let refusal = format!("call=\"{call}\" by=operator-a code={code} ");
        assert!(
            line.contains("membership call refused") && line.contains(&refusal),
            "{line}"
        );
    }
    assert_eq!(
        held.load(Ordering::Relaxed),
        0,
        "a line arrived with the turn held: {logged}"
    );
}
