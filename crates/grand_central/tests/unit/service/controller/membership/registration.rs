//! The cases of a registration of one row: the row an add writes and the
//! adds that are refused, the life a replacement and an upsert make or keep,
//! the status a replacement carries, the place an answer carries, the
//! bootstrap port a row states, and a request or a row that is refused by
//! name.

use super::*;

#[test]
fn a_registration_writes_the_row_its_caller_stated_and_answers_it_pending() {
    let (store, door) = door();
    let told = door
        .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
        .expect("registered");
    let id = numbered(URL, 0);
    let stored = store.get(id).expect("the row is in the store");
    assert_eq!(
        told,
        proto::WorkerEvent {
            seq: 2,
            membership_epoch: 1,
            event: Some(Event::Registered(super::super::worker_wire::row(&stored))),
            term: TERM,
        }
    );
    let row = row_of(&told);
    assert_eq!(
        (row.id, row.incarnation, row.revision, row.status),
        (id.0, life(TERM, 0), 1, proto::WorkerStatus::Pending as i32)
    );
    assert_eq!(
        (stored.status, stored.revision, store.epoch()),
        (WorkerStatus::Pending, 1, 1)
    );
    // No mode reads as an add.
    let other = "grpc://10.0.0.6:8000";
    let unset = proto::RegisterWorkerRequest {
        spec: Some(spec(other)),
        ..Default::default()
    };
    let told = door
        .register(&operator(), under(TERM), &unset)
        .expect("registered");
    assert!(matches!(told.event, Some(Event::Registered(_))));
    assert_eq!((told.membership_epoch, told.seq, told.term), (2, 3, TERM));
}

#[test]
fn a_second_add_is_refused_and_writes_nothing() {
    let (store, door) = door();
    door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
        .expect("registered");
    let refused = door
        .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
        .expect_err("the URL stands");
    assert_eq!(refused.code(), Code::AlreadyExists);
    assert_eq!(
        refused.message(),
        format!(
            "the URL is registered as worker {:016x} at revision 1",
            numbered(URL, 0).0
        )
    );
    assert!(!refused.message().contains("10.0.0.5"));
    assert_eq!(store.epoch(), 1);
}

/// A call reads the row that stands off the worker objects, and no
/// row of the table. An add over a row that a writer outside the
/// holder left in the table is decided over no object. The store
/// refuses it, the holder lays nothing, and the call is answered as a
/// row that moved, the add again too. A cut laid whole lays the row,
/// and the add is then refused as an add over any row that stands.
/// None writes.
#[test]
fn an_add_over_a_row_written_around_the_holder_is_refused_and_lays_nothing() {
    // A row that names its engine, as every writer of the crate lays
    // one, and a row that names none: its ranks are asked under its
    // key, and it has none.
    for engine in [URL, ""] {
        let (store, door) = door();
        let facts = WorkerFacts {
            key: URL.to_owned(),
            engine: engine.to_owned(),
            incarnation: 1,
            wire_modes: WireModes::GRPC,
            models: vec!["model-a".to_owned()],
            ..WorkerFacts::default()
        };
        store.upsert_by(facts, Registrant::Control);
        let stood = (store.epoch(), store.by_key(URL));
        let refused = |code: Code, why: &str| {
            let refused = door
                .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
                .expect_err(why);
            assert_eq!(refused.code(), code, "engine {engine:?}: {why}");
            assert_eq!((store.epoch(), store.by_key(URL)), stood);
        };
        for _ in 0..2 {
            refused(Code::Aborted, "decided over no object");
            assert!(door.objects.is_empty(), "a refused add lays nothing");
        }
        // The holder's pass over a cut's rows, run over the store's:
        // this store leads a term and refuses a restore of its own cut.
        lay_standing(&door.objects, &store);
        refused(Code::AlreadyExists, "the URL stands");
        assert_eq!(door.objects.len(), 1, "the cut laid it");
    }
}

#[test]
fn a_replace_wants_a_standing_row_and_makes_its_next_incarnation() {
    let (store, door) = door();
    let absent = door
        .register(&operator(), under(TERM), &ask(spec(URL), Mode::Replace))
        .expect_err("nothing to replace");
    assert_eq!(absent.code(), Code::NotFound);
    // What a new row names is no rule of a call that makes none: a
    // REPLACE that leaves the wire and the role unsaid is not found
    // either, and so is one that leaves the role alone unsaid.
    let bare = proto::WorkerSpec {
        url: URL.to_owned(),
        models: vec!["model-a".to_owned()],
        ..Default::default()
    };
    let roleless = proto::WorkerSpec {
        wire: proto::WireMode::Grpc as i32,
        ..bare.clone()
    };
    for unsaid in [bare.clone(), roleless] {
        let absent = door
            .register(&operator(), under(TERM), &ask(unsaid, Mode::Replace))
            .expect_err("nothing to replace");
        assert_eq!(
            (absent.code(), absent.message()),
            (Code::NotFound, "no worker is registered under the URL")
        );
    }
    // A field past its bounds is refused as itself, found or not, and
    // whatever the call leaves unsaid: a field read ahead of the wire
    // (the models) and one read after it (the labels), each with the
    // wire and the role stated and with both left unsaid.
    let wide = |over: proto::WorkerSpec| proto::WorkerSpec {
        models: vec!["m".repeat(257)],
        ..over
    };
    let labelled = |over: proto::WorkerSpec| proto::WorkerSpec {
        labels: (0..129)
            .map(|n| (format!("label-{n}"), String::new()))
            .collect(),
        ..over
    };
    let models = "spec.models: a model id is at most 256 bytes";
    let labels = "spec.labels: a row holds at most 128 labels";
    for (past, field) in [
        (wide(spec(URL)), models),
        (wide(bare.clone()), models),
        (labelled(spec(URL)), labels),
        (labelled(bare.clone()), labels),
    ] {
        let refused = door
            .register(&operator(), under(TERM), &ask(past, Mode::Replace))
            .expect_err("past a bound");
        assert_eq!(
            (refused.code(), refused.message()),
            (Code::InvalidArgument, field)
        );
    }
    // An UPSERT of the same would make a row, and is held to its names.
    let unnamed = door
        .register(&operator(), under(TERM), &ask(bare, Mode::Upsert))
        .expect_err("a new row names its wire");
    assert_eq!(unnamed.code(), Code::InvalidArgument);
    assert_eq!(store.epoch(), 0);

    door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
        .expect("registered");
    let old = store.get(numbered(URL, 0)).expect("the first life");
    door.connect_row(old.id, old.revision, TERM);
    let old = store.get(old.id).expect("the first life, ready");

    let told = door
        .register(&operator(), under(TERM), &ask(spec(URL), Mode::Replace))
        .expect("replaced");
    // Numbered at the epoch the two writes above left.
    let new = store.get(numbered(URL, 2)).expect("the second life");
    assert_eq!(store.get(old.id), None);
    assert_eq!(
        told.event,
        Some(Event::Replaced(proto::Replaced {
            old: Some(super::super::worker_wire::row(&old)),
            new: Some(super::super::worker_wire::row(&new)),
            // The same role, wire and models: the status came over.
            state_carried: true,
        }))
    );
    assert_eq!((new.status, new.revision), (WorkerStatus::Ready, 1));
}

#[test]
fn many_short_model_ids_register_and_a_list_past_the_row_size_is_refused() {
    let (store, door) = door();
    let heavy = proto::WorkerSpec {
        models: vec!["m".repeat(256); 64],
        ..spec(URL)
    };
    let status = door
        .register(&operator(), under(TERM), &ask(heavy, Mode::Add))
        .expect_err("past the row's size");
    assert_eq!(
        (status.code(), status.message()),
        (
            Code::InvalidArgument,
            "spec: a row is at most 16384 bytes on the wire"
        )
    );
    assert_eq!(store.epoch(), 0);

    let short = proto::WorkerSpec {
        models: (0..129).map(|n| n.to_string()).collect(),
        ..spec(URL)
    };
    let wrote = door
        .register(&operator(), under(TERM), &ask(short.clone(), Mode::Add))
        .expect("the ids fit");
    assert_eq!(row_of(&wrote).models, short.models);
    assert_eq!(
        store.by_key(URL).expect("the row").facts.models,
        short.models
    );
}

#[test]
fn a_replacement_says_carried_only_where_the_status_came_over() {
    let (store, door) = door();
    let waited = |spec, mode| proto::RegisterWorkerRequest {
        wait_ready: true,
        ..ask(spec, mode)
    };
    door.register(&operator(), under(TERM), &waited(unchecked(URL), Mode::Add))
        .expect("registered");
    // The same role, wire and models: the ready status came over.
    let told = door
        .register(
            &operator(),
            under(TERM),
            &waited(unchecked(URL), Mode::Replace),
        )
        .expect("replaced");
    let Some(Event::Replaced(same)) = told.event else {
        panic!("a replacement: {told:?}");
    };
    assert!(same.state_carried);
    // Another model: a fresh row, ready by its own connect signal, so
    // nothing came over.
    let other = proto::WorkerSpec {
        models: vec!["model-b".to_owned()],
        ..unchecked(URL)
    };
    let told = door
        .register(&operator(), under(TERM), &waited(other, Mode::Replace))
        .expect("replaced");
    let Some(Event::Replaced(fresh)) = told.event else {
        panic!("a replacement: {told:?}");
    };
    assert!(!fresh.state_carried);
    assert_eq!(
        fresh.new.map(|new| new.status),
        Some(proto::WorkerStatus::Ready as i32)
    );
    assert_eq!(
        store.by_key(URL).map(|row| row.status),
        Some(WorkerStatus::Ready)
    );
}

#[test]
fn an_upsert_of_equal_facts_writes_nothing_and_answers_the_row_unmoved() {
    let (store, door) = door();
    door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Upsert))
        .expect("registered");
    let stood = store.get(numbered(URL, 0)).expect("the row");
    for _ in 0..3 {
        let told = door
            .register(&operator(), under(TERM), &ask(spec(URL), Mode::Upsert))
            .expect("answered");
        assert_eq!(
            told,
            proto::WorkerEvent {
                seq: 2,
                membership_epoch: 1,
                event: Some(Event::Relabelled(super::super::worker_wire::row(&stood))),
                term: TERM,
            }
        );
    }
    assert_eq!(store.epoch(), 1);
    assert_eq!(store.get(stood.id), Some(stood));
}

#[test]
fn an_upsert_changes_a_standing_row_in_place_or_makes_a_new_incarnation() {
    let (store, door) = door();
    door.register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
        .expect("registered");
    let first = numbered(URL, 0);

    // A label: the same worker, its facts moved.
    let labelled = proto::WorkerSpec {
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        ..spec(URL)
    };
    let told = door
        .register(
            &operator(),
            under(TERM),
            &ask(labelled.clone(), Mode::Upsert),
        )
        .expect("relabelled");
    assert!(matches!(told.event, Some(Event::Relabelled(_))));
    let row = row_of(&told);
    assert_eq!(
        (row.id, row.incarnation, row.revision),
        (first.0, life(TERM, 0), 2)
    );
    assert_eq!(row.labels.get("tier").map(String::as_str), Some("gold"));
    assert_eq!(told.membership_epoch, 2);

    // The role: another worker under the URL, so a new incarnation.
    let decode = proto::WorkerSpec {
        role: proto::Role::Decode as i32,
        ..labelled
    };
    let told = door
        .register(&operator(), under(TERM), &ask(decode.clone(), Mode::Upsert))
        .expect("replaced");
    let Some(Event::Replaced(replaced)) = told.event else {
        panic!("a change of role is a replacement: {told:?}");
    };
    assert!(!replaced.state_carried);
    let new = replaced.new.expect("the new row");
    assert_eq!(
        (new.id, new.incarnation, new.revision),
        (numbered(URL, 2).0, life(TERM, 2), 1)
    );
    assert_eq!(replaced.old.map(|old| old.id), Some(first.0));
    assert_eq!(store.get(first), None);

    // The models alone, then the wire alone, then what the worker
    // states for a request's hard demands alone, then whether it takes
    // images alone: each is another worker under the URL too.
    let remodelled = proto::WorkerSpec {
        models: vec!["model-b".to_owned()],
        ..decode
    };
    let rewired = proto::WorkerSpec {
        wire: proto::WireMode::Http as i32,
        ..remodelled.clone()
    };
    let stating = proto::WorkerSpec {
        demand_facts: Some(proto::DemandFacts {
            runtime: 7,
            ..Default::default()
        }),
        ..rewired.clone()
    };
    let seeing = proto::WorkerSpec {
        vision: Some(true),
        ..stating.clone()
    };
    let changes = [
        ("models", remodelled),
        ("wire", rewired),
        ("demand facts", stating.clone()),
        ("whether it takes images", seeing.clone()),
    ];
    for (what, changed) in changes {
        let stood = store.by_key(URL).expect("the row");
        let told = door
            .register(&operator(), under(TERM), &ask(changed, Mode::Upsert))
            .expect("replaced");
        let Some(Event::Replaced(replaced)) = told.event else {
            panic!("a change of {what} is a replacement: {told:?}");
        };
        assert_eq!(replaced.old.map(|old| old.id), Some(stood.id.0), "{what}");
        assert_eq!(store.get(stood.id), None, "{what}");
    }

    // The row holds the block and the bit, and the same registration
    // again finds equal facts: it writes nothing. Neither does one that
    // states the block and leaves the bit out, since a bit left out
    // keeps the row's.
    let stood = store.by_key(URL).expect("the row");
    assert_eq!(stood.facts.demand_facts.runtime, 7);
    assert!(stood.facts.vision);
    for again in [seeing, stating] {
        let told = door
            .register(&operator(), under(TERM), &ask(again, Mode::Upsert))
            .expect("answered");
        assert!(matches!(told.event, Some(Event::Relabelled(_))));
        assert_eq!(store.by_key(URL).as_ref(), Some(&stood));
    }
}

#[test]
fn an_upsert_of_the_same_models_in_another_order_is_made_in_place() {
    let (store, door) = door();
    let listing = |models: &[&str]| proto::WorkerSpec {
        models: models.iter().map(|&model| model.to_owned()).collect(),
        ..unchecked(URL)
    };
    let two = listing(&["model-a", "model-b"]);
    let told = door
        .register(&operator(), under(TERM), &ask(two.clone(), Mode::Add))
        .expect("registered");
    let first = row_of(&told);
    // Unchecked, the row is ready by its connect signal: a state to
    // carry.
    assert_eq!(
        store.by_key(URL).map(|row| row.status),
        Some(WorkerStatus::Ready)
    );
    // The same two in the other order, then one of them named twice:
    // the same worker each time, and the row tells the stated list.
    for stated in [
        &["model-b", "model-a"][..],
        &["model-a", "model-b", "model-a"][..],
    ] {
        let told = door
            .register(
                &operator(),
                under(TERM),
                &ask(listing(stated), Mode::Upsert),
            )
            .expect("reordered");
        let Some(Event::Relabelled(row)) = told.event else {
            panic!("a reorder is made in place: {told:?}");
        };
        assert_eq!((row.id, row.incarnation), (first.id, first.incarnation));
        assert_eq!(row.models, stated);
        assert_eq!(row.status, proto::WorkerStatus::Ready as i32);
    }
    // A replacement that names the same set carries the status too.
    let told = door
        .register(&operator(), under(TERM), &ask(two, Mode::Replace))
        .expect("replaced");
    let Some(Event::Replaced(replaced)) = told.event else {
        panic!("a replacement: {told:?}");
    };
    assert!(replaced.state_carried);
}

#[test]
fn a_wait_for_a_checked_worker_and_a_request_out_of_order_are_refused_by_name() {
    let (store, door) = door();
    let waited = proto::RegisterWorkerRequest {
        wait_ready: true,
        ..ask(spec(URL), Mode::Add)
    };
    let bare = proto::RegisterWorkerRequest::default();
    let unknown = proto::RegisterWorkerRequest {
        mode: 9,
        ..ask(spec(URL), Mode::Add)
    };
    let wide = ask(
        proto::WorkerSpec {
            dp_size: 65,
            ..spec(URL)
        },
        Mode::Add,
    );
    for (request, path) in [
        (waited, "wait_ready: "),
        (bare, "spec: "),
        (unknown, "mode: "),
        (wide, "spec.dp_size: "),
    ] {
        let refused = door
            .register(&operator(), under(TERM), &request)
            .expect_err("refused");
        assert_eq!(refused.code(), Code::InvalidArgument, "{path}");
        assert!(
            refused.message().starts_with(path),
            "{path}: {}",
            refused.message()
        );
    }
    assert_eq!(store.epoch(), 0);
}

#[test]
fn an_answer_carries_the_place_of_its_calls_write_under_that_places_term() {
    let (store, door) = door();
    // Not waited: the registration as its write left it. The connect
    // signal's write is the next place.
    let told = door
        .register(&operator(), under(TERM), &ask(unchecked(URL), Mode::Add))
        .expect("registered");
    assert_eq!((told.seq, told.term, told.membership_epoch), (2, TERM, 1));
    assert_eq!(row_of(&told).status, proto::WorkerStatus::Pending as i32);
    // Waited: the row after its connect signal, at that write's place.
    let other = "grpc://10.0.0.6:8000";
    let waited = proto::RegisterWorkerRequest {
        wait_ready: true,
        ..ask(unchecked(other), Mode::Add)
    };
    let told = door
        .register(&operator(), under(TERM), &waited)
        .expect("registered");
    assert_eq!((told.seq, told.term, told.membership_epoch), (5, TERM, 4));
    assert_eq!(row_of(&told).status, proto::WorkerStatus::Ready as i32);
    // A call that writes nothing answers the place the log stands at:
    // here the start of the next term.
    store::LeasesStore::set_term(store.as_ref(), TERM + 1);
    let told = door
        .register(
            &operator(),
            under(TERM + 1),
            &ask(unchecked(other), Mode::Upsert),
        )
        .expect("answered");
    assert_eq!(
        (told.seq, told.term, told.membership_epoch),
        (6, TERM + 1, 4)
    );
    // And the next write is made under that term.
    let drain = proto::DeregisterWorkerRequest {
        worker_id: row_of(&told).id,
        revision: row_of(&told).revision,
        drain: true,
    };
    let told = door
        .deregister(&operator(), under(TERM + 1), &drain)
        .expect("draining");
    assert_eq!(
        (told.seq, told.term, told.membership_epoch),
        (7, TERM + 1, 5)
    );
}

#[test]
fn a_row_past_a_bound_is_refused_by_name_and_nothing_is_written() {
    fn labelled(count: usize) -> proto::WorkerSpec {
        let labels = (0..count).map(|n| (n.to_string(), "v".repeat(4_096)));
        proto::WorkerSpec {
            labels: labels.collect(),
            ..spec(URL)
        }
    }
    let (store, door) = door();
    // A registration: by its field, and by its spec for the row whole.
    let long = ask(spec(&"u".repeat(2_049)), Mode::Add);
    let heavy = ask(labelled(4), Mode::Add);
    for (request, path) in [(long, "spec.url: "), (heavy, "spec: ")] {
        let refused = door
            .register(&operator(), under(TERM), &request)
            .expect_err("refused");
        assert_eq!(refused.code(), Code::InvalidArgument, "{path}");
        assert!(
            refused.message().starts_with(path),
            "{path}: {}",
            refused.message()
        );
    }
    assert_eq!(store.epoch(), 0);
    // A patch: by its labels, for the row it would make. The row stands.
    let row = row_of(
        &door
            .register(&operator(), under(TERM), &ask(labelled(3), Mode::Add))
            .expect("registered"),
    );
    let fourth = proto::PatchWorkerRequest {
        worker_id: row.id,
        labels: [("3".to_owned(), "v".repeat(4_096))].into(),
        revision: row.revision,
        ..Default::default()
    };
    let refused = door
        .patch(&operator(), under(TERM), &fourth)
        .expect_err("refused");
    assert_eq!(refused.code(), Code::InvalidArgument);
    assert!(
        refused.message().starts_with("labels: "),
        "{}",
        refused.message()
    );
    let stands = store.get(numbered(URL, 0)).expect("the row");
    assert_eq!(
        (stands.revision, stands.facts.labels.len(), store.epoch()),
        (1, 3, 1)
    );
}

#[test]
fn a_registration_states_its_bootstrap_port_and_a_change_of_it_is_made_in_place() {
    let (store, door) = door();
    let with = |bootstrap_port| proto::WorkerSpec {
        pairing: Some(proto::PairingDescriptor {
            bootstrap_port,
            ..Default::default()
        }),
        ..spec(URL)
    };
    let port =
        |told: &proto::WorkerEvent| row_of(told).pairing.map(|pairing| pairing.bootstrap_port);
    let told = door
        .register(&operator(), under(TERM), &ask(with(8_998), Mode::Add))
        .expect("registered");
    let first = row_of(&told);
    assert_eq!(port(&told), Some(8_998));
    assert_eq!(
        store.by_key(URL).and_then(|row| row.facts.bootstrap_port),
        Some(8_998)
    );

    // Another port: the same worker, its facts moved.
    let told = door
        .register(&operator(), under(TERM), &ask(with(9_001), Mode::Upsert))
        .expect("relabelled");
    assert!(matches!(told.event, Some(Event::Relabelled(_))));
    let moved = row_of(&told);
    assert_eq!((moved.id, moved.revision), (first.id, first.revision + 1));
    assert_eq!(port(&told), Some(9_001));

    // A registration that states no descriptor leaves the port: equal
    // facts, nothing written.
    let told = door
        .register(&operator(), under(TERM), &ask(spec(URL), Mode::Upsert))
        .expect("answered");
    assert_eq!(port(&told), Some(9_001));
    assert_eq!((told.membership_epoch, store.epoch()), (2, 2));

    // A REPLACE that states no descriptor is the worker's next
    // incarnation and keeps the port that stands.
    let told = door
        .register(&operator(), under(TERM), &ask(spec(URL), Mode::Replace))
        .expect("replaced");
    let next = row_of(&told);
    assert_ne!(next.id, first.id, "a life of its own");
    assert_eq!(port(&told), Some(9_001));

    // Removed and registered again, the worker tells no port.
    let removal = proto::DeregisterWorkerRequest {
        worker_id: next.id,
        revision: next.revision,
        drain: false,
    };
    door.deregister(&operator(), under(TERM), &removal)
        .expect("removed");
    let told = door
        .register(&operator(), under(TERM), &ask(spec(URL), Mode::Add))
        .expect("registered again");
    assert_eq!(row_of(&told).pairing, None);

    // A worker registered with no port tells no descriptor.
    let other = "grpc://10.0.0.6:8000";
    let told = door
        .register(&operator(), under(TERM), &ask(spec(other), Mode::Add))
        .expect("registered");
    assert_eq!(row_of(&told).pairing, None);
}
