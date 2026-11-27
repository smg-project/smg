//! What a registration expects under its URLs: an add over an absent URL,
//! a replacement over the row its asker saw, and what refuses each. A
//! found row's event signal, stated by the door itself.

use super::*;

#[test]
fn a_registration_is_laid_over_what_it_expects() {
    check_expectations(local_door);
}

#[test]
fn a_found_row_states_its_event_signal() {
    check_event_signal(local_door);
}

/// A generation whose prefill profile of `pd/*` reads the index, and no
/// other profile of it.
fn prefill_reads_the_index() -> AppliedSpec {
    let text = r#"{
          "apiVersion": "smg.policy/v1",
          "profiles": {
            "default": { "filter": ["healthy"], "pick": "round-robin" },
            "prefill": { "filter": ["healthy"], "score": ["prefix-affinity"], "pick": "best" },
            "decode": { "filter": ["healthy"], "score": ["least-load"], "pick": "best" }
          },
          "bindings": [
            { "match": { "model": "pd/*", "phase": ["prefill", "decode"] }, "plan": "pd-pair", "profiles": { "prefill": "prefill", "decode": "decode" } },
            { "match": { "model": "*" }, "profile": "default" }
          ]
        }"#;
    let mut spec: PolicySpec = serde_json::from_str(text).expect("a well-formed document");
    spec.generation = 1;
    AppliedSpec::validate(&spec).expect("the document validates")
}

/// The door states a found row's event signal itself, whatever the
/// facts state: what the generation in force reads for the row's first
/// model in the phase of its role, and only for a gRPC worker. A write
/// under a later generation states it again.
pub(crate) fn check_event_signal(open: impl Fn() -> (Arc<MemoryStore>, Box<dyn Rows>)) {
    const PAIRED: &str = "pd/x";
    // The row's model, role and wire, what its facts state, and what the
    // door states.
    let rows = [
        (PAIRED, Role::Prefill, WireModes::GRPC, false, true),
        (PAIRED, Role::Decode, WireModes::GRPC, true, false),
        (PAIRED, Role::Single, WireModes::GRPC, true, false),
        (PAIRED, Role::Prefill, WireModes::HTTP, true, false),
        ("other", Role::Prefill, WireModes::GRPC, true, false),
    ];
    for (model, role, wire_modes, events, asked) in rows {
        let (store, door) = open();
        store
            .apply(prefill_reads_the_index())
            .expect("the first generation");
        let found = WorkerFacts {
            models: vec![model.to_owned()],
            role,
            wire_modes,
            signals: Signals {
                events,
                ..Signals::default()
            },
            ..stated()
        };
        door.found_one_over(&Registrant::Control, Precondition::Absent, URL, &|_| {
            found.clone()
        })
        .expect("registered");
        let row = store.by_key(URL).expect("the found row");
        assert_eq!(
            row.facts.signals.events, asked,
            "{model} {role:?} {wire_modes:?}"
        );
        assert_eq!(indexed_under(&row).is_some(), asked);

        // Every profile of the cache-aware preset reads the index.
        let mut later = PolicySpec::preset("cache_aware").expect("the preset");
        later.generation = 2;
        store
            .apply(AppliedSpec::validate(&later).expect("the preset validates"))
            .expect("the second generation");
        door.found_one(&Registrant::Control, URL, &|_| found.clone())
            .expect("laid again");
        let row = store.by_key(URL).expect("the row laid again");
        assert_eq!(
            row.facts.signals.events,
            wire_modes == WireModes::GRPC,
            "{model} {role:?} {wire_modes:?} under the later generation"
        );
    }
}

/// An add registers an absent URL. Where every row it states stands it
/// is refused before a fact is stated, whoever made the rows. A
/// replacement is laid over the row its asker saw by the rule every
/// registration takes, and is refused where that row moved or left. An
/// add of ranks over a standing rank that differs writes no rank.
pub(crate) fn check_expectations(open: Open) {
    let (store, door) = open();
    let control = Registrant::Control;
    let pool = Registrant::provider("pool", "private/key").expect("a maker");
    let unstated = |_: Option<&WorkerFacts>| -> WorkerFacts { panic!("no fact is stated") };

    let added = door.found_one_over(&control, Precondition::Absent, URL, &|_| stated());
    let row = store.by_key(URL).expect("the added row");
    assert_eq!(added, Ok(MembershipWriteOutcome::Registered(row.id)));
    let epoch = store.epoch();
    for maker in [&control, &pool] {
        assert_eq!(
            door.found_one_over(maker, Precondition::Absent, URL, &unstated),
            Err(MembershipWriteRefused::Stands)
        );
    }

    let seen = Precondition::AtRevision(standing(&row));
    assert_eq!(
        door.found_one_over(&control, seen, URL, &|_| stated()),
        Ok(MembershipWriteOutcome::Unmoved(row.id))
    );
    assert_eq!(
        door.found_one_over(&pool, seen, URL, &unstated),
        Err(MembershipWriteRefused::NotRegistrant(
            "the row was made by a control call: its maker alone changes it".to_owned()
        ))
    );
    assert_eq!(store.epoch(), epoch);

    // New labels are laid in place, and the row's revision moves on.
    let relabelled = |_: Option<&WorkerFacts>| WorkerFacts {
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        ..stated()
    };
    assert_eq!(
        door.found_one_over(&control, seen, URL, &relabelled),
        Ok(MembershipWriteOutcome::InPlace(row.id))
    );
    assert_eq!(
        door.found_one_over(&control, seen, URL, &unstated),
        Err(MembershipWriteRefused::Moved)
    );

    // Other models are the URL's next life, as under an upsert.
    let moved = store.by_key(URL).expect("the relabelled row");
    let other = |_: Option<&WorkerFacts>| WorkerFacts {
        models: vec!["other".to_owned()],
        ..stated()
    };
    let wrote = door.found_one_over(
        &control,
        Precondition::AtRevision(standing(&moved)),
        URL,
        &other,
    );
    let next = store.by_key(URL).expect("the next life");
    assert_eq!(
        wrote,
        Ok(MembershipWriteOutcome::Another {
            old: row.id,
            new: next.id
        })
    );

    // A draining row is leaving: its removal wins over a replacement,
    // which is told so. Another maker is told the row is not its own.
    let seen = standing(&next);
    assert_eq!(
        door.drained(Remover::Registrant(&control), URL, seen),
        Ok(MembershipWriteOutcome::Draining(next.id))
    );
    assert_eq!(
        door.found_one_over(&control, Precondition::AtRevision(seen), URL, &unstated),
        Err(MembershipWriteRefused::Draining)
    );
    assert!(matches!(
        door.found_one_over(&pool, Precondition::AtRevision(seen), URL, &unstated),
        Err(MembershipWriteRefused::NotRegistrant(_))
    ));
    assert_eq!(
        store.by_key(URL).map(|row| (row.id, row.status)),
        Some((next.id, WorkerStatus::Draining))
    );

    // The row its asker saw left.
    assert_eq!(
        door.gone(Remover::Registrant(&control), URL, seen),
        Ok(MembershipWriteOutcome::Removed(next.id))
    );
    assert_eq!(
        door.found_one_over(&control, Precondition::AtRevision(seen), URL, &unstated),
        Err(MembershipWriteRefused::Absent)
    );
    assert_eq!(store.by_key(URL), None);

    // The second of two ranks stands with another label: the add
    // writes neither. Both ranks are registered at the door and the
    // first is removed there, so the second stands alone.
    let (store, door) = open();
    let two = ranked(2);
    let gold = WorkerFacts {
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        ..two[1].clone()
    };
    find_rows(&*door, &control, &[two[0].clone(), gold]).expect("two ranks");
    let first = store.by_key(&two[0].key).expect("rank zero");
    door.gone(Remover::Registrant(&control), &two[0].key, standing(&first))
        .expect("rank zero removed");
    let epoch = store.epoch();
    assert_eq!(
        find_rows_over(&*door, &control, Precondition::Absent, &two),
        Err(MembershipWriteRefused::Stands)
    );
    assert_eq!(store.epoch(), epoch);

    // Two ranks are added in one write. The engine's row in its other
    // form refuses an add, and a replacement states one row.
    let (store, door) = open();
    let add = |facts: &[WorkerFacts]| find_rows_over(&*door, &control, Precondition::Absent, facts);
    let ranks = add(&two).expect("two ranks added");
    assert!(matches!(
        ranks[..],
        [
            MembershipWriteOutcome::Registered(_),
            MembershipWriteOutcome::Registered(_)
        ]
    ));
    let epoch = store.epoch();
    assert_eq!(add(&[stated()]), Err(MembershipWriteRefused::Stands));
    let rank = store.by_key(&two[0].key).expect("rank zero");
    assert!(matches!(
        find_rows_over(
            &*door,
            &control,
            Precondition::AtRevision(standing(&rank)),
            &two
        ),
        Err(MembershipWriteRefused::Past(FactPastBound {
            fact: "rows",
            ..
        }))
    ));
    assert_eq!(store.epoch(), epoch);
    find_rows(&*door, &control, &[stated()]).expect("plain again");
    let epoch = store.epoch();
    assert_eq!(add(&two), Err(MembershipWriteRefused::Stands));
    assert_eq!(store.epoch(), epoch);
}
