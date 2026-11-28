//! The cases of a registration of several ranks: the row it writes a rank,
//! the modes over an engine, a registration of one rank over ranks that
//! stand, a lead lost inside the call, the same add called again, the bound
//! of 64 ranks, and the entries of its answer.

use super::*;

/// A row that names no engine dials its key, and its engine's ranks
/// are asked under that key. Here such a row stands beside a rank that
/// names the key as its engine, both written through the holder's own
/// calls with facts no call of the door states: a
/// plain registration of the URL is refused as over any engine that
/// stands as ranks.
#[test]
fn a_row_that_names_no_engine_is_asked_for_its_ranks_under_its_key() {
    let (store, door) = door();
    let plain = WorkerFacts {
        key: URL.to_owned(),
        engine: String::new(),
        incarnation: 1,
        wire_modes: WireModes::GRPC,
        models: vec!["model-a".to_owned()],
        ..WorkerFacts::default()
    };
    let rank = WorkerFacts {
        key: format!("{URL}@1"),
        engine: URL.to_owned(),
        dp_rank: 1,
        dp_size: 2,
        ..plain.clone()
    };
    for facts in [plain, rank] {
        let laid = door.objects.upsert(&store, facts, Registrant::Control);
        laid.expect("a row through the holder");
    }
    let refused = door
        .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
        .expect_err("its engine stands as ranks");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::FailedPrecondition, super::super::AS_RANKS)
    );
}

/// A registration that states several ranks writes one row a rank under
/// the rank's URL and answers each as its write left it, at the one
/// place of the call's last write. Stated again it writes nothing and
/// answers each rank unmoved.
#[test]
fn a_registration_of_several_ranks_writes_a_row_a_rank_and_answers_each() {
    use proto::rank_row::Made;
    let (store, door) = door();
    let told = door
        .register(&operator(), under(TERM), &ask(wide(URL, 3), Mode::Add))
        .expect("registered");
    assert_eq!((told.membership_epoch, told.seq, told.term), (3, 4, TERM));
    let made = made_of(&told);
    assert_eq!(made.len(), 3);
    for (rank, made) in (0..3_u16).zip(&made) {
        let stored = store
            .by_key(&format!("{URL}@{rank}"))
            .expect("the rank's row");
        assert_eq!(
            *made,
            Made::Registered(super::super::worker_wire::row(&stored))
        );
        let facts = &stored.facts;
        assert_eq!(
            (facts.engine.as_str(), facts.dp_rank, facts.dp_size),
            (URL, rank, 3)
        );
        assert_eq!((stored.status, stored.revision), (WorkerStatus::Pending, 1));
    }

    let again = door
        .register(&operator(), under(TERM), &ask(wide(URL, 3), Mode::Upsert))
        .expect("unmoved");
    assert_eq!(
        (again.membership_epoch, again.seq, store.epoch()),
        (3, 4, 3)
    );
    for (rank, made) in made_of(&again).iter().enumerate() {
        let stored = store
            .by_key(&format!("{URL}@{rank}"))
            .expect("the rank's row");
        assert_eq!(
            *made,
            Made::Relabelled(super::super::worker_wire::row(&stored))
        );
    }
}

/// The modes hold for the engine. An add is refused where its ranks
/// stand or the engine stands in another form. A replacement wants every
/// stated rank and makes each its next life. An upsert lays each rank,
/// and the engine's rows it does not state leave with it.
#[test]
fn the_modes_of_a_registration_of_several_ranks_hold_for_the_engine() {
    use proto::rank_row::Made;
    let (store, door) = door();
    let register =
        |spec: proto::WorkerSpec, mode| door.register(&operator(), under(TERM), &ask(spec, mode));
    let rank = |rank: u16| store.by_key(&format!("{URL}@{rank}"));

    let none = register(wide(URL, 2), Mode::Replace).expect_err("no rank stands");
    assert_eq!(none.code(), Code::NotFound);
    register(spec(URL), Mode::Add).expect("the plain row");
    let other_form = register(wide(URL, 2), Mode::Add).expect_err("the plain row stands");
    assert_eq!(other_form.code(), Code::AlreadyExists);
    assert_eq!(store.epoch(), 1);

    // A worker that comes back ranked loses its plain row.
    let plain = store.by_key(URL).expect("the plain row");
    let told = register(wide(URL, 2), Mode::Upsert).expect("ranked");
    let made = made_of(&told);
    assert!(
        matches!(
            made.as_slice(),
            [Made::Registered(_), Made::Registered(_), Made::Removed(left)]
                if left.worker_id == plain.id.0
        ),
        "{made:?}"
    );
    assert!(store.by_key(URL).is_none());
    assert_eq!((store.epoch(), told.membership_epoch), (4, 4));

    let stands = register(wide(URL, 2), Mode::Add).expect_err("a rank stands");
    assert_eq!(stands.code(), Code::AlreadyExists);
    let short = register(wide(URL, 3), Mode::Replace).expect_err("rank two is absent");
    assert_eq!(short.code(), Code::NotFound);
    assert_eq!(store.epoch(), 4);

    // A replacement is the next life of each rank.
    let old = [0, 1].map(|at| rank(at).expect("a rank"));
    let told = register(wide(URL, 2), Mode::Replace).expect("replaced");
    for (old, made) in old.iter().zip(made_of(&told)) {
        let Made::Replaced(replaced) = made else {
            panic!("{made:?}");
        };
        assert_eq!(replaced.old.map(|row| row.id), Some(old.id.0));
        assert_ne!(replaced.new.map(|row| row.id), Some(old.id.0));
        assert!(!replaced.state_carried);
    }

    // Wider, then narrower: the rank it no longer states leaves.
    let told = register(wide(URL, 3), Mode::Upsert).expect("wider");
    let made = made_of(&told);
    assert!(
        matches!(
            made.as_slice(),
            [
                Made::Relabelled(_),
                Made::Relabelled(_),
                Made::Registered(_)
            ]
        ),
        "{made:?}"
    );
    let third = rank(2).expect("rank two");
    let told = register(wide(URL, 2), Mode::Upsert).expect("narrower");
    let made = made_of(&told);
    assert!(
        matches!(
            made.as_slice(),
            [Made::Relabelled(_), Made::Relabelled(_), Made::Removed(left)]
                if left.worker_id == third.id.0
        ),
        "{made:?}"
    );
    assert!(rank(2).is_none());
    assert_eq!(rank(0).map(|row| row.facts.dp_size), Some(2));
}

/// A registration of one rank for an engine that stands as ranks is
/// refused by name in every mode and writes nothing: a plain row beside
/// its ranks would be the engine twice. The way off is a rank at a time.
#[test]
fn a_registration_of_one_rank_over_an_engine_that_stands_as_ranks_is_refused() {
    let (store, door) = door();
    door.register(&operator(), under(TERM), &ask(wide(URL, 2), Mode::Add))
        .expect("ranks");
    // Under the engine's URL and under a rank's own: one rank laid
    // alone would offer what the engine's other rank does not.
    let rank = format!("{URL}@0");
    for mode in [Mode::Add, Mode::Replace, Mode::Upsert] {
        for url in [URL, rank.as_str()] {
            let refused = door
                .register(&operator(), under(TERM), &ask(spec(url), mode))
                .expect_err("the engine stands as ranks");
            assert_eq!(
                (refused.code(), refused.message()),
                (Code::FailedPrecondition, super::super::AS_RANKS),
                "{mode:?} {url}"
            );
        }
    }
    assert_eq!(store.epoch(), 2);

    for at in 0..2 {
        let row = store.by_key(&format!("{URL}@{at}")).expect("a rank");
        let leave = proto::DeregisterWorkerRequest {
            worker_id: row.id.0,
            revision: row.revision,
            drain: false,
        };
        door.deregister(&operator(), under(TERM), &leave)
            .expect("the rank leaves");
    }
    door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
        .expect("the plain row");
    assert!(store.by_key(URL).is_some());
}

/// A one-rank replacement or upsert over ranks another maker laid is
/// told who made them, ahead of the engine's form, as the wide call
/// tells it. An add asks no maker.
#[test]
fn a_registration_of_one_rank_over_another_makers_ranks_is_told_the_maker_first() {
    let (store, door) = door();
    door.upsert(
        &operator(),
        &found(URL),
        TERM,
        &ask(wide(URL, 2), Mode::Add),
    )
    .expect("a provider's ranks");
    let rank = format!("{URL}@0");
    let register = |spec: proto::WorkerSpec, mode| {
        door.register(&operator(), under(TERM), &ask(spec, mode))
            .expect_err("ranks of another maker")
    };
    for mode in [Mode::Replace, Mode::Upsert] {
        for asked in [spec(URL), spec(&rank), wide(URL, 2)] {
            let url = asked.url.clone();
            let refused = register(asked, mode);
            assert_eq!(refused.code(), Code::PermissionDenied, "{mode:?} {url}");
        }
    }
    for url in [URL, rank.as_str()] {
        let refused = register(spec(url), Mode::Add);
        assert_eq!(
            (refused.code(), refused.message()),
            (Code::FailedPrecondition, super::super::AS_RANKS),
            "{url}"
        );
    }
    assert_eq!(store.epoch(), 2);
}

/// A wide call that loses its lead between two ranks is refused. The
/// rank written before the refusal stands, and its write leaves its
/// audit line ahead of the refusal's.
#[test]
fn a_term_lost_between_two_ranks_leaves_the_first_rank_and_its_audit_line() {
    let lines = Logged::default();
    let writer = lines.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let _beside = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let _kept = tracing::subscriber::set_default(subscriber);

    let (_, feed) = journaled_from(&LeasePlace::Process, 0);
    let journal = Arc::new(LosesTheLead {
        log: Arc::clone(&feed.log),
        holds: AtomicUsize::new(0),
    });
    let store = Arc::new(MemoryStore::journaled(journal).with_decision_slots(1));
    store::LeasesStore::set_term(store.as_ref(), TERM);
    let log = Arc::clone(&feed.log);
    let door = MembershipCalls::based(Arc::clone(&store), log, 0, holder(&store));

    let refused = door
        .register(&operator(), under(TERM), &ask(wide(URL, 2), Mode::Add))
        .expect_err("the lead was lost at the second rank");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::Unavailable, super::super::LEADS_NO_TERM)
    );
    assert!(store.by_key(&format!("{URL}@0")).is_some());
    assert!(store.by_key(&format!("{URL}@1")).is_none());

    let logged = String::from_utf8(lines.0.lock().clone()).expect("the lines are text");
    let lines: Vec<&str> = logged.lines().collect();
    let [written, refusal] = lines.as_slice() else {
        panic!("one write's line and the refusal's: {logged}");
    };
    assert!(
        written.contains("membership write") && written.contains("made=\"registered\""),
        "{written}"
    );
    assert!(refusal.contains("membership call refused"), "{refusal}");
    assert!(!logged.contains(URL), "{logged}");
}

/// A door whose journal loses the lead when told, with `width` ranks of
/// [`URL`] registered under a lead that holds.
fn losing(width: u32) -> (Arc<MemoryStore>, MembershipCalls, Arc<LosesTheLead>) {
    let (_, feed) = journaled_from(&LeasePlace::Process, 0);
    let journal = Arc::new(LosesTheLead {
        log: Arc::clone(&feed.log),
        holds: AtomicUsize::new(usize::MAX),
    });
    let handed = Arc::clone(&journal);
    let store = Arc::new(MemoryStore::journaled(handed).with_decision_slots(1));
    store::LeasesStore::set_term(store.as_ref(), TERM);
    let log = Arc::clone(&feed.log);
    let door = MembershipCalls::based(Arc::clone(&store), log, 0, holder(&store));
    door.register(&operator(), under(TERM), &ask(wide(URL, width), Mode::Add))
        .expect("the ranks registered under a held lead");
    (store, door, journal)
}

/// A lead lost behind a rank's write in place refuses the next rank's.
/// The term is asked ahead of every write, whatever it makes of its
/// row.
#[test]
fn a_term_lost_behind_a_write_in_place_stops_the_next_ranks_write() {
    let (store, door, journal) = losing(2);
    journal.holds.store(0, Ordering::Relaxed);
    let relabelled = proto::WorkerSpec {
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        ..wide(URL, 2)
    };
    let refused = door
        .register(&operator(), under(TERM), &ask(relabelled, Mode::Upsert))
        .expect_err("the lead was lost at the second rank");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::Unavailable, super::super::LEADS_NO_TERM)
    );
    let labels = |rank: u32| {
        let row = store.by_key(&format!("{URL}@{rank}")).expect("the rank");
        row.facts.labels.len()
    };
    assert_eq!((labels(0), labels(1)), (1, 0));
}

/// A lead lost behind the last rank that stays refuses the removal of
/// the rank a narrower registration no longer states.
#[test]
fn a_term_lost_ahead_of_a_removal_leaves_the_rank_that_would_leave() {
    let (store, door, journal) = losing(3);
    journal.holds.store(1, Ordering::Relaxed);
    let refused = door
        .register(&operator(), under(TERM), &ask(wide(URL, 2), Mode::Upsert))
        .expect_err("the lead was lost ahead of the removal");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::Unavailable, super::super::LEADS_NO_TERM)
    );
    let width = |rank: u32| {
        store
            .by_key(&format!("{URL}@{rank}"))
            .map(|row| row.facts.dp_size)
    };
    assert_eq!((width(0), width(1), width(2)), (Some(2), Some(2), Some(3)));
}

/// A door whose lead was lost behind the first of an add's three ranks
/// and came back under a later term: the next leader's door, over the
/// row the first attempt left.
fn led_again_behind_a_lost_add() -> (Arc<MemoryStore>, MembershipCalls) {
    let (_, feed) = journaled_from(&LeasePlace::Process, 0);
    let journal = Arc::new(LosesTheLead {
        log: Arc::clone(&feed.log),
        holds: AtomicUsize::new(0),
    });
    let store = Arc::new(MemoryStore::journaled(journal).with_decision_slots(1));
    store::LeasesStore::set_term(store.as_ref(), TERM);
    let log = Arc::clone(&feed.log);
    let door = MembershipCalls::based(Arc::clone(&store), log, 0, holder(&store));
    let lost = door
        .register(&operator(), under(TERM), &ask(wide(URL, 3), Mode::Add))
        .expect_err("the lead was lost at the second rank");
    assert_eq!(
        (lost.code(), lost.message()),
        (Code::Unavailable, super::super::LEADS_NO_TERM)
    );
    assert_eq!((store.epoch(), store.all_rows().1.len()), (1, 1));
    store::LeasesStore::set_term(store.as_ref(), TERM + 2);
    (store, door)
}

/// An add that lost its lead behind its first rank is the same add at
/// the next leader's door. The rank it wrote answers unmoved, the
/// others are registered, and each rank has its one row. Called once
/// more, the engine stands whole: the add is refused by the first
/// rank's row.
#[test]
fn an_add_called_again_behind_a_lost_lead_lays_the_ranks_it_left() {
    use proto::rank_row::Made;
    let (store, door) = led_again_behind_a_lost_add();
    let first = store.by_key(&format!("{URL}@0")).expect("the first rank");
    let add = || door.register(&operator(), under(TERM + 2), &ask(wide(URL, 3), Mode::Add));

    let told = add().expect("the same add");
    let made = made_of(&told);
    assert!(
        matches!(
            made.as_slice(),
            [Made::Relabelled(stood), Made::Registered(_), Made::Registered(_)]
                if stood.id == first.id.0
        ),
        "{made:?}"
    );
    assert_eq!(store.by_key(&format!("{URL}@0")).as_ref(), Some(&first));
    assert_eq!((store.epoch(), store.all_rows().1.len()), (3, 3));

    let whole = add().expect_err("the engine stands whole");
    assert_eq!(whole.code(), Code::AlreadyExists);
    assert_eq!(
        whole.message(),
        format!(
            "the URL is registered as worker {:016x} at revision {}",
            first.id.0, first.revision
        )
    );
    assert_eq!(store.epoch(), 3);
}

/// The same add with another label than the standing rank's is refused
/// by that rank's row and writes nothing: an add lays the rest by what
/// stands alone.
#[test]
fn an_add_called_again_with_another_label_is_refused_and_writes_nothing() {
    let (store, door) = led_again_behind_a_lost_add();
    let labelled = proto::WorkerSpec {
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        ..wide(URL, 3)
    };
    let refused = door
        .register(&operator(), under(TERM + 2), &ask(labelled, Mode::Add))
        .expect_err("the first rank stands with other labels");
    assert_eq!(refused.code(), Code::AlreadyExists);
    assert_eq!((store.epoch(), store.all_rows().1.len()), (1, 1));
}

/// A registration states at most 64 ranks. At the bound, with every rank
/// replaced by rows near their size bound, the answer stays inside a
/// 4 MiB frame: a replacement tells two rows.
#[test]
fn a_registration_states_at_most_64_ranks_and_its_widest_answer_fits_a_frame() {
    use prost::Message as _;
    let (store, door) = door();
    let past = door
        .register(&operator(), under(TERM), &ask(wide(URL, 65), Mode::Add))
        .expect_err("past the bound");
    assert_eq!(
        (past.code(), past.message()),
        (
            Code::InvalidArgument,
            "spec.dp_size: a registration states at most 64 ranks"
        )
    );
    assert_eq!(store.epoch(), 0);

    let heavy = proto::WorkerSpec {
        models: vec!["m".repeat(256); 60],
        ..wide(URL, 64)
    };
    door.register(&operator(), under(TERM), &ask(heavy.clone(), Mode::Add))
        .expect("64 ranks");
    let told = door
        .register(&operator(), under(TERM), &ask(heavy, Mode::Replace))
        .expect("64 ranks replaced");
    let made = made_of(&told);
    assert_eq!(made.len(), 64);
    assert!(made
        .iter()
        .all(|made| matches!(made, proto::rank_row::Made::Replaced(_))));
    let bytes = told.encoded_len();
    assert!(bytes > 1 << 20, "{bytes}");
    assert!(bytes < 4 << 20, "{bytes}");
}

/// Each entry of the answer is the event a watcher is told of that row:
/// the log's entries of one call, as a watch tells them, read beside
/// the answer. A rank whose checks are off is answered as its write left
/// it, and its connect signal is the watcher's next event. The answer's
/// place is the call's last write.
#[test]
fn each_entry_of_an_answer_of_several_ranks_is_the_event_a_watcher_is_told() {
    use proto::rank_row::Made;
    let (_store, door) = door();
    let watched = |after: Position| {
        let mut entries = Vec::new();
        door.log
            .read(after, 64, &mut entries)
            .expect("the log holds the call's writes");
        entries
            .iter()
            .filter_map(super::super::worker_wire::member)
            .collect::<Vec<_>>()
    };
    let entries = |told: &proto::WorkerEvent| match told.event.clone().expect("an event") {
        Event::Ranks(ranks) => ranks.rows,
        other => panic!("no ranks in {other:?}"),
    };
    let ranked = |mode| {
        let spec = proto::WorkerSpec {
            dp_size: 2,
            ..unchecked(URL)
        };
        let before = door.place();
        let told = door
            .register(&operator(), under(TERM), &ask(spec, mode))
            .expect("the ranks are laid");
        let events = watched(before);
        let last = events.last().expect("the call's last write");
        assert_eq!(
            (told.seq, told.membership_epoch, told.term),
            (last.seq, last.membership_epoch, last.term)
        );
        // The connect signals are the watcher's own events.
        let rows: Vec<_> = events
            .iter()
            .filter_map(|event| event.event.clone())
            .filter(|event| !matches!(event, Event::Status(_)))
            .collect();
        let answered = entries(&told);
        assert_eq!(rows.len(), answered.len());
        for (event, entry) in rows.into_iter().zip(&answered) {
            assert_eq!(
                super::super::worker_wire::rank_row(event).as_ref(),
                Some(entry)
            );
        }
        answered
    };

    // A worker that comes back ranked: two rows registered, the plain
    // row removed.
    door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
        .expect("the plain row");
    let made = ranked(Mode::Upsert);
    assert!(
        matches!(
            made.iter()
                .map(|entry| entry.made.as_ref())
                .collect::<Vec<_>>()[..],
            [
                Some(Made::Registered(_)),
                Some(Made::Registered(_)),
                Some(Made::Removed(_))
            ]
        ),
        "{made:?}"
    );
    // A replacement of ranks that are ready hands their status on.
    let made = ranked(Mode::Replace);
    assert_eq!(made.len(), 2);
    for entry in &made {
        let Some(Made::Replaced(replaced)) = &entry.made else {
            panic!("{entry:?}");
        };
        assert!(replaced.state_carried, "{replaced:?}");
    }
}
