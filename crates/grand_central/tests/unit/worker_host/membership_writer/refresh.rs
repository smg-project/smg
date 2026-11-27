//! A refresh: the labels of its rebuilt worker and a pairing descriptor,
//! written over the row its asker saw. It states no other fact and
//! starts no life.

use super::*;

/// A worker a registration or a refresh built at [`URL`], under one
/// label.
fn built(tier: &str) -> BasicWorkerBuilder {
    <BasicWorkerBuilder>::new(URL)
        .model(ModelCard::new("m"))
        .label("tier", tier)
}

#[test]
fn a_refresh_writes_its_workers_labels_and_no_other_fact() {
    check_refresh(local_door).expect("the makers and their rows");
}

/// A refresh writes the labels its rebuilt worker states over the row
/// its asker saw, whoever made the row. The row keeps its maker and
/// every other fact, and equal labels write no revision. A worker that
/// states another fact is refused in the refresh's own variant, a label
/// past its bound is refused, and a row that moved, drains or left
/// takes no refresh. A refused refresh writes nothing.
pub(crate) fn check_refresh(
    open: impl Fn() -> (Arc<MemoryStore>, Box<dyn Rows>),
) -> Result<(), &'static str> {
    let provider = Registrant::provider("pool", "private/key").map_err(|_| "the source's maker")?;
    let worker = |built: BasicWorkerBuilder| -> Arc<dyn Worker> { Arc::new(built.build()) };
    for maker in [&Registrant::Control, &provider] {
        let (store, door) = open();
        let first = worker(built("gold"));
        assert!(door
            .found_built(maker, Precondition::Absent, &first, None)
            .is_ok());
        let row = store.by_key(URL).ok_or("the registered row")?;
        let seen = standing(&row);

        let equal = door.refreshed(seen, &worker(built("gold")), &first);
        assert_eq!(equal, Ok(MembershipWriteOutcome::Unmoved(row.id)));

        // Each states one fact more than a label: a model, a role, a
        // health threshold.
        let checks = HealthCheckConfig {
            failure_threshold: 9,
            ..HealthCheckConfig::default()
        };
        let strays = [
            built("silver").model(ModelCard::new("other")),
            built("silver").worker_type(WorkerType::Prefill),
            built("silver").health_config(checks),
        ];
        for stray in strays {
            let refused = door.refreshed(seen, &worker(stray), &first);
            assert_eq!(refused, Err(MembershipWriteRefused::NotLabelsAlone));
        }
        let wide = "x".repeat(WorkerFacts::LABEL_VALUE_BYTES + 1);
        let refused = door.refreshed(seen, &worker(built(&wide)), &first);
        assert!(
            matches!(
                refused,
                Err(MembershipWriteRefused::Past(FactPastBound {
                    fact: "labels",
                    ..
                }))
            ),
            "{refused:?}"
        );
        assert_eq!(store.by_key(URL).as_ref(), Some(&row));

        let wrote = door.refreshed(seen, &worker(built("silver")), &first);
        assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));
        let moved = store.by_key(URL).ok_or("the refreshed row")?;
        let mut facts = row.facts.clone();
        facts.labels.insert("tier".to_owned(), "silver".to_owned());
        assert_eq!(
            moved,
            WorkerRow {
                facts,
                revision: row.revision + 1,
                ..row
            }
        );

        // The asker's row moved under it, then drains, then is gone.
        let late = worker(built("bronze"));
        assert_eq!(
            door.refreshed(seen, &late, &first),
            Err(MembershipWriteRefused::Moved)
        );
        let seen = standing(&moved);
        let by = Remover::Registrant(maker);
        assert_eq!(
            door.drained(by, URL, seen),
            Ok(MembershipWriteOutcome::Draining(moved.id))
        );
        assert_eq!(
            door.refreshed(seen, &late, &first),
            Err(MembershipWriteRefused::Draining)
        );
        assert_eq!(
            door.gone(by, URL, seen),
            Ok(MembershipWriteOutcome::Removed(moved.id))
        );
        assert_eq!(
            door.refreshed(seen, &late, &first),
            Err(MembershipWriteRefused::Absent)
        );
    }
    Ok(())
}

/// A refresh's scope is its worker's labels and the pairing descriptor
/// they derive. A descriptor that moved is laid in place, with a label
/// or by itself, and the row keeps its id, its maker and every other
/// fact. One that differs from the row's by its blanks alone is the
/// row's. One that reports nothing is none, stated or held: it
/// withdraws a row's descriptor and writes nothing over a row with
/// none. Another fact beside a moved descriptor is refused, and
/// nothing is written.
#[test]
fn a_refresh_lays_the_pairing_its_labels_derive() {
    let (store, door) = door();
    let first: Arc<dyn Worker> = Arc::new(built("gold").build());
    let registered = door.found_built(&Registrant::Control, Precondition::Absent, &first, None);
    assert!(registered.is_ok(), "{registered:?}");
    let row = store.by_key(URL).expect("the registered row");
    let version = |version: &str| {
        Some(Arc::new(PairingDescriptor {
            version: Some(version.to_owned()),
            ..PairingDescriptor::default()
        }))
    };
    // A refresh whose facts are the row's own asks no word of the row's
    // object. Each ask is counted, and answered no.
    let asked = std::cell::Cell::new(0);
    let no_object = |_: &WorkerRow, _: &WorkerFacts| {
        asked.set(asked.get() + 1);
        false
    };
    let refresh = |seen: &WorkerRow, pairing: Option<Arc<PairingDescriptor>>, tier: &str| {
        let state = |standing: &WorkerFacts| {
            let mut facts = standing.clone();
            facts.pairing.clone_from(&pairing);
            facts.labels.insert("tier".to_owned(), tier.to_owned());
            facts
        };
        refresh_stated(&door, standing(seen), URL, &state, &no_object)
    };

    // The descriptor and a label move together.
    let wrote = refresh(&row, version("2"), "silver");
    assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));
    let moved = store.by_key(URL).expect("the refreshed row");
    let mut facts = row.facts.clone();
    facts.labels.insert("tier".to_owned(), "silver".to_owned());
    facts.pairing = version("2");
    assert_eq!(
        moved,
        WorkerRow {
            facts,
            revision: row.revision + 1,
            ..row.clone()
        }
    );

    // The same version between blanks is the row's descriptor.
    let equal = refresh(&moved, version(" 2 "), "silver");
    assert_eq!(equal, Ok(MembershipWriteOutcome::Unmoved(row.id)));
    assert_eq!(store.by_key(URL).as_ref(), Some(&moved));

    // The descriptor moves by itself, then is withdrawn: by none, and
    // by one that reports nothing, which the row holds as none.
    let empty = || Some(Arc::new(PairingDescriptor::default()));
    let mut bare = moved.clone();
    for withdrawn in [None, version(" ")] {
        let wrote = refresh(&bare, version("3"), "silver");
        assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));
        let again = store.by_key(URL).expect("the row of a moved descriptor");
        assert_eq!(again.facts.pairing, version("3"));
        assert_eq!(again.facts.labels, moved.facts.labels);
        let wrote = refresh(&again, withdrawn, "silver");
        assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));
        bare = store
            .by_key(URL)
            .expect("the row of a withdrawn descriptor");
        assert_eq!(bare.facts.pairing, None);
    }

    // Over a row with none, one that reports nothing writes nothing.
    for nothing in [empty(), version(" ")] {
        let unmoved = refresh(&bare, nothing, "silver");
        assert_eq!(unmoved, Ok(MembershipWriteOutcome::Unmoved(row.id)));
        assert_eq!(store.by_key(URL).as_ref(), Some(&bare));
    }

    // Another fact beside a moved descriptor is outside the scope: the
    // first of these refreshes to ask a word of the row's object.
    assert_eq!(asked.get(), 0);
    let other = |standing: &WorkerFacts| {
        let mut facts = standing.clone();
        facts.pairing = version("4");
        facts.priority += 1;
        facts
    };
    let stray = refresh_stated(&door, standing(&bare), URL, &other, &no_object);
    assert_eq!(stray, Err(MembershipWriteRefused::NotLabelsAlone));
    assert_eq!(asked.get(), 1);
    assert_eq!(store.by_key(URL).as_ref(), Some(&bare));

    // A row may hold one that reports nothing, as a registration states
    // it. It reads as a row with none.
    let laid = door.found(&Registrant::Control, Precondition::Any, URL, &|standing| {
        let mut facts = standing.expect("the standing row").clone();
        facts.pairing = empty();
        facts
    });
    assert_eq!(laid, Ok(MembershipWriteOutcome::InPlace(row.id)));
    let held = store.by_key(URL).expect("the row of the registration");
    assert_eq!(held.facts.pairing, empty());
    for nothing in [None, empty()] {
        let unmoved = refresh(&held, nothing, "silver");
        assert_eq!(unmoved, Ok(MembershipWriteOutcome::Unmoved(row.id)));
        assert_eq!(store.by_key(URL).as_ref(), Some(&held));
    }
}

/// A reading that states the worker's own descriptor, as a host that
/// keeps one reads a worker: the door's reading, and the descriptor the
/// worker's labels derive.
fn with_its_descriptor(
    worker: &dyn Worker,
    standing: Option<&WorkerFacts>,
    over: Option<Over<'_>>,
) -> WorkerFacts {
    WorkerFacts {
        pairing: found::pairing_descriptor(worker),
        ..found::stated(worker, standing, over)
    }
}

/// A row may state a fact that its object holds in a narrower form: a
/// part of a second, a threshold of zero, a runtime's name as it was
/// written. A refresh's worker is rebuilt from the row, so it states the
/// object's form. The door holds it against the row's own object: its
/// labels and the descriptor they derive may move, and the row keeps its
/// own word of every other fact. A worker that states a fact as neither the
/// row nor its object holds it is refused. So is one that states the
/// object's seconds beside the row's threshold, and one built with its
/// checks off over a row that states probes. An upsert of the row's
/// object still states the object's form.
#[test]
fn a_refresh_keeps_the_rows_word_of_a_fact_its_object_holds_narrower() {
    use smg_worker::worker::RuntimeType;

    use super::super::super::object::{builder, unchecked};

    let (store, door) = door();
    let version = |version: &str| {
        Some(Arc::new(PairingDescriptor {
            version: Some(version.to_owned()),
            ..PairingDescriptor::default()
        }))
    };
    let narrower = WorkerFacts {
        engine_kind: "VLLM".to_owned(),
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        // A descriptor that no label of the row derives.
        pairing: version("1"),
        health: HealthPolicy {
            failure_threshold: 0,
            success_threshold: 0,
            check_interval_ms: 2_500,
            timeout_ms: 1_500,
            probes: true,
        },
        ..stated()
    };
    let laid = door.found(&Registrant::Control, Precondition::Absent, URL, &|_| {
        narrower.clone()
    });
    assert!(laid.is_ok(), "{laid:?}");
    let row = store.by_key(URL).expect("the registered row");
    assert_eq!(
        (&row.facts.engine_kind, row.facts.health),
        (&narrower.engine_kind, narrower.health)
    );
    // The row's own object: three seconds and two, a threshold of one
    // twice, and the runtime the worker crate reads the name as.
    let first = crate::worker_host::object(&row);
    let held = first.metadata();
    let checks = &held.health_config;
    assert_eq!(
        (
            checks.check_interval_secs,
            checks.timeout_secs,
            checks.failure_threshold,
            checks.success_threshold,
            held.spec.runtime_type,
        ),
        (3, 2, 1, 1, RuntimeType::Vllm)
    );
    // What a refresh rebuilds: the row under other labels.
    let relabelled = |seen: &WorkerRow, labels: &[(&str, &str)]| WorkerRow {
        facts: WorkerFacts {
            labels: labels
                .iter()
                .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
                .collect(),
            ..seen.facts.clone()
        },
        ..seen.clone()
    };

    // The row's own object states nothing new.
    let unmoved = door.refreshed(standing(&row), &first, &first);
    assert_eq!(unmoved, Ok(MembershipWriteOutcome::Unmoved(row.id)));

    // A label moves, and the row keeps its word.
    let silver = relabelled(&row, &[("tier", "silver")]);
    let built = crate::worker_host::object(&silver);
    let wrote = door.refreshed(standing(&row), &built, &first);
    assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));
    let moved = store.by_key(URL).expect("the refreshed row");
    assert_eq!(moved.facts, silver.facts);

    // A label moves the descriptor it derives too, in place: the row
    // keeps its word there as well. The row's own labels derive none, and
    // the row's descriptor stands in its object's reading.
    let versioned = relabelled(&moved, &[("tier", "silver"), ("version", "2")]);
    let built = crate::worker_host::object(&versioned);
    let wrote = refresh(&door, standing(&moved), built.as_ref(), with_its_descriptor);
    assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));
    let paired = store.by_key(URL).expect("the row of a moved descriptor");
    assert_eq!(
        paired.facts,
        WorkerFacts {
            pairing: version("2"),
            ..versioned.facts.clone()
        }
    );

    // A fact as neither the row nor its object holds it is outside the
    // scope: a timeout of nine seconds, another runtime, the checks off.
    // So is the object's form beside the row's threshold of zero.
    let bronze = relabelled(&paired, &[("tier", "bronze")]);
    let slower = HealthCheckConfig {
        timeout_secs: 9,
        ..checks.clone()
    };
    let mixed = HealthCheckConfig {
        failure_threshold: 0,
        ..checks.clone()
    };
    let other = WorkerRow {
        facts: WorkerFacts {
            engine_kind: "sglang".to_owned(),
            ..bronze.facts.clone()
        },
        ..bronze.clone()
    };
    let strays: [Arc<dyn Worker>; 4] = [
        Arc::new(builder(&bronze).health_config(slower).build()),
        crate::worker_host::object(&other),
        Arc::new(unchecked(&bronze).build()),
        Arc::new(builder(&bronze).health_config(mixed).build()),
    ];
    for stray in strays {
        let refused = door.refreshed(standing(&paired), &stray, &built);
        assert_eq!(refused, Err(MembershipWriteRefused::NotLabelsAlone));
        assert_eq!(store.by_key(URL).as_ref(), Some(&paired));
    }

    // An upsert of the row's own object states the object's form, under
    // the row's labels.
    let rebuilt = crate::worker_host::object(&paired);
    let upserted = door.found_built(&Registrant::Control, Precondition::Any, &rebuilt, None);
    assert!(upserted.is_ok(), "{upserted:?}");
    let own = store.by_key(URL).expect("the upserted row").facts;
    assert_eq!(
        (own.engine_kind.as_str(), own.health, &own.labels),
        (
            "vllm",
            HealthPolicy {
                failure_threshold: 1,
                success_threshold: 1,
                check_interval_ms: 3_000,
                timeout_ms: 2_000,
                probes: true,
            },
            &paired.facts.labels
        )
    );
}

/// A row may state a threshold of zero, which its object holds as one. A
/// worker that states the zero states the row itself, and one rebuilt from
/// the row states the object's one. A refresh takes each under another
/// label, and the row keeps its zero.
#[test]
fn a_refresh_takes_the_rows_own_word_where_its_object_holds_less() {
    let (store, door) = door();
    let zero = HealthCheckConfig {
        failure_threshold: 0,
        success_threshold: 0,
        ..HealthCheckConfig::default()
    };
    let first: Arc<dyn Worker> = Arc::new(built("gold").health_config(zero.clone()).build());
    let registered = door.found_built(&Registrant::Control, Precondition::Absent, &first, None);
    assert!(registered.is_ok(), "{registered:?}");
    let row = store.by_key(URL).expect("the registered row");
    let thresholds = |row: &WorkerRow| {
        let policy = row.facts.health;
        (policy.failure_threshold, policy.success_threshold)
    };
    assert_eq!(thresholds(&row), (0, 0));
    let own = crate::worker_host::object(&row);
    let held = &own.metadata().health_config;
    assert_eq!((held.failure_threshold, held.success_threshold), (1, 1));

    // A worker that states the zero states the row.
    let silver: Arc<dyn Worker> = Arc::new(built("silver").health_config(zero).build());
    let wrote = door.refreshed(standing(&row), &silver, &first);
    assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));
    let moved = store.by_key(URL).expect("the refreshed row");
    let tier = |row: &WorkerRow| row.facts.labels.get("tier").cloned();
    assert_eq!(tier(&moved).as_deref(), Some("silver"));
    assert_eq!(thresholds(&moved), (0, 0));

    // One rebuilt from the row states the object's one.
    let bronze = WorkerRow {
        facts: WorkerFacts {
            labels: [("tier".to_owned(), "bronze".to_owned())].into(),
            ..moved.facts.clone()
        },
        ..moved.clone()
    };
    let rebuilt = crate::worker_host::object(&bronze);
    let wrote = door.refreshed(standing(&moved), &rebuilt, &silver);
    assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));
    let again = store.by_key(URL).expect("the row refreshed again");
    assert_eq!(again.facts, bronze.facts);
}

/// A wait wider than a row holds reads as the widest number a row holds.
/// A row less than 295 milliseconds under that number builds an object
/// that waits past it. A refresh takes that object under another label,
/// and the row keeps its number. A worker that waits a second longer for
/// either check reads as that object does, and passes as it.
#[test]
fn a_wait_past_a_rows_width_passes_a_refresh_as_the_rows_object() {
    use super::super::super::object::builder;

    let (store, door) = door();
    let widest = WorkerFacts {
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        health: HealthPolicy {
            failure_threshold: 3,
            success_threshold: 2,
            check_interval_ms: 4_294_967_001,
            timeout_ms: 4_294_967_001,
            probes: true,
        },
        ..stated()
    };
    let laid = door.found(&Registrant::Control, Precondition::Absent, URL, &|_| {
        widest.clone()
    });
    assert!(laid.is_ok(), "{laid:?}");
    let row = store.by_key(URL).expect("the registered row");
    assert_eq!(row.facts.health, widest.health);
    // The row's own object waits 4,294,968 seconds for each check: its
    // reading stands at the widest number.
    let first = crate::worker_host::object(&row);
    let checks = first.metadata().health_config.clone();
    assert_eq!(
        (checks.check_interval_secs, checks.timeout_secs),
        (4_294_968, 4_294_968)
    );
    let read = |worker: &dyn Worker| found::facts(worker, Some(&row.facts)).health;
    let own = read(first.as_ref());
    assert_eq!(
        (own.check_interval_ms, own.timeout_ms),
        (u32::MAX, u32::MAX)
    );
    let relabelled = |seen: &WorkerRow, tier: &str| WorkerRow {
        facts: WorkerFacts {
            labels: [("tier".to_owned(), tier.to_owned())].into(),
            ..seen.facts.clone()
        },
        ..seen.clone()
    };

    // The row's object under another label moves the label, and the row
    // keeps its number.
    let silver = relabelled(&row, "silver");
    let mut over = crate::worker_host::object(&silver);
    let wrote = door.refreshed(standing(&row), &over, &first);
    assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));
    let mut seen = store.by_key(URL).expect("the refreshed row");
    assert_eq!(seen.facts, silver.facts);

    // A second longer for one check reads as the row's object does, and
    // the refresh takes each: the label moves, and the number stays. Each
    // worker is stated twice: the seconds it adds to the row's object's,
    // the interval's then the timeout's, and the seconds it then waits.
    let slower = [
        ("bronze", (0, 1), (4_294_968, 4_294_969)),
        ("copper", (1, 0), (4_294_969, 4_294_968)),
    ];
    for (tier, (interval, timeout), waited) in slower {
        let next = relabelled(&seen, tier);
        let waits = HealthCheckConfig {
            check_interval_secs: checks.check_interval_secs + interval,
            timeout_secs: checks.timeout_secs + timeout,
            ..checks.clone()
        };
        let longer: Arc<dyn Worker> = Arc::new(builder(&next).health_config(waits).build());
        let held = &longer.metadata().health_config;
        assert_eq!((held.check_interval_secs, held.timeout_secs), waited);
        assert_eq!(read(longer.as_ref()), own);
        let wrote = door.refreshed(standing(&seen), &longer, &over);
        assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));
        seen = store.by_key(URL).expect("the row refreshed again");
        assert_eq!(seen.facts, next.facts);
        over = longer;
    }
}
