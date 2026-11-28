//! The cases of a provider's writes through the replica's own writer: the
//! checks every membership writer passes, the facts a row may state there,
//! a write where the replica leads and where it does not, what a label
//! refresh writes of a row, and the answer the two doors give of one row.

use super::*;

/// A standby builds its rows from the log. A fact the log's row does not
/// carry would leave it another row than its leader holds.
#[test]
fn a_fact_the_log_does_not_carry_is_refused_by_its_name_and_nothing_is_written() {
    use super::super::super::worker_wire;
    use crate::{
        engine::DemandFacts,
        store::{pairing::PairingDescriptor, EncodeHandoff, EncodeOffer},
    };
    type State = fn(&mut WorkerFacts);

    let (store, door) = door();
    let hosted = ReplicaMembershipWriter::new(Arc::new(door), operator(), under(TERM));
    let maker = found("pods/a");
    let epoch = store.epoch();
    let lost: [(&str, State); 7] = [
        ("row.pairing", |facts| {
            facts.pairing = Some(Arc::new(PairingDescriptor {
                protocol: Some("pd".to_owned()),
                ..PairingDescriptor::default()
            }));
        }),
        // An empty connector, which the wire's row reads as none.
        ("row.encode", |facts| {
            facts.encode = EncodeOffer {
                handoff: EncodeHandoff::Push,
                producer: true,
                connector: Some(String::new()),
                ..EncodeOffer::default()
            };
        }),
        // A connector's one bound is the row's.
        ("row", |facts| {
            facts.encode = EncodeOffer {
                handoff: EncodeHandoff::Push,
                producer: true,
                connector: Some("a".repeat(WorkerFacts::ROW_BYTES)),
                ..EncodeOffer::default()
            };
        }),
        ("row.zone", |facts| facts.zone = Some(String::new())),
        ("row.max_running", |facts| facts.max_running = Some(0)),
        ("row.kv_block_size", |facts| facts.kv_block_size = Some(0)),
        ("row.bootstrap_port", |facts| facts.bootstrap_port = Some(0)),
    ];
    for (fact, state) in lost {
        let answer = hosted.found_one(&maker, URL, &|_| {
            let mut facts = found_facts();
            state(&mut facts);
            facts
        });
        let Err(MembershipWriteRefused::Past(past)) = answer else {
            panic!("{fact}: wrong answer: {answer:?}")
        };
        assert_eq!(past.fact, fact);
    }
    assert_eq!((store.by_key(URL), store.epoch()), (None, epoch));

    // What the log carries is taken, and a standby reads it back whole:
    // what the worker states for a request's hard demands among it,
    // whether it takes images, its end of the encode handoff, its priority
    // and a drain window of its own. The offer's connector is the store's
    // text, held to no rule of its words: one that holds a blank and a
    // control character, past a name's bound.
    let stated = DemandFacts {
        runtime: 7,
        references: true,
        host_pipeline: true,
        window: 32_768,
        decoding: 3,
    };
    let offered = EncodeOffer {
        handoff: EncodeHandoff::Push,
        producer: true,
        connector: Some(format!("connector a\u{7}{}", "b".repeat(256))),
        ..EncodeOffer::default()
    };
    let stating = |_: Option<&WorkerFacts>| WorkerFacts {
        demand_facts: stated,
        vision: true,
        encode: offered.clone(),
        priority: 5,
        drain_settle_secs: Some(30),
        ..found_facts()
    };
    hosted.found_one(&maker, URL, &stating).expect("found");
    let row = store.by_key(URL).expect("the found row");
    assert_eq!(row.facts.demand_facts, stated);
    assert!(row.facts.vision);
    assert_eq!(row.facts.encode, offered);
    let stated = (row.facts.priority, row.facts.drain_settle_secs);
    assert_eq!(stated, (5, Some(30)));
    let back = worker_wire::told_row(&worker_wire::row(&row), std::time::Instant::now());
    assert_eq!(back.map(|back| back.facts), Ok(row.facts));
}

/// The store reads a row as host-local by its reach alone: a row of the
/// host-local wire that names no gateway would stand in every gateway's
/// pool. A store held in process serves one gateway.
#[test]
fn a_host_local_wire_names_its_gateway_at_the_controllers_door() {
    let stated = |wire: WireModes, from: Reachability| {
        move |_: Option<&WorkerFacts>| WorkerFacts {
            wire_modes: wire,
            reachable_from: from,
            ..found_facts()
        }
    };
    let (store, door) = door();
    let hosted = ReplicaMembershipWriter::new(Arc::new(door), operator(), under(TERM));
    let maker = found("pods/a");
    let epoch = store.epoch();
    for wire in [WireModes::ZMQ, WireModes::GRPC.union(WireModes::ZMQ)] {
        let answer = hosted.found_one(&maker, URL, &stated(wire, Reachability::Any));
        let Err(MembershipWriteRefused::Past(past)) = answer else {
            panic!("{wire:?}: wrong answer: {answer:?}")
        };
        assert_eq!(past.fact, "row.wire");
    }
    assert_eq!((store.by_key(URL), store.epoch()), (None, epoch));

    // A row that names its gateway is in that gateway's pool alone.
    let one = Reachability::Gateway(GatewayId(7));
    hosted
        .found_one(&maker, URL, &stated(WireModes::ZMQ, one))
        .expect("found");
    let row = store.by_key(URL).expect("the found row");
    assert_eq!(row.facts.reachable_from, one);
}

#[test]
fn a_registration_is_laid_over_what_it_expects() {
    check_expectations(hosted_door);
}

#[test]
fn a_found_row_states_its_event_signal() {
    check_event_signal(hosted_door);
}

#[test]
fn a_hosted_door_checks_and_writes_the_whole_registration() {
    check_rank_rows(hosted_door).expect("rank registrations at the hosted door");
}

#[test]
fn a_hosted_door_keeps_the_write_of_one_row() {
    check_one_row(hosted_door);
}

#[test]
fn a_removal_takes_only_the_row_its_asker_saw() {
    check_guard(hosted_door).expect("both registrations stand in turn");
}

#[test]
fn a_refresh_writes_its_workers_labels_and_no_other_fact() {
    check_refresh(hosted_door).expect("the makers and their rows");
}

#[test]
fn the_hosts_own_source_removes_its_rows_under_any_key() {
    check_provider(hosted_door).expect("the makers and their rows");
}

#[test]
fn health_removes_inside_its_scope_alone() {
    check_health_scope(hosted_door).expect("the makers and their rows");
}

#[test]
fn a_removal_leaves_the_rows_maker_as_it_was() {
    check_maker(hosted_door).expect("the maker and its row");
}

#[test]
fn a_providers_write_on_a_replica_that_does_not_lead_is_refused_and_writes_nothing() {
    use std::sync::atomic::AtomicUsize;

    let (store, door) = door();
    let door = Arc::new(door);
    let maker = found("pods/a");
    ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM))
        .found_one(&maker, URL, &|_| found_facts())
        .expect("registered");
    let row = store.by_key(URL);
    let seen = standing(row.as_ref().expect("the registered row"));
    let by = Remover::Registrant(&maker);
    let epoch = store.epoch();
    let at = door.place();
    let reads = Arc::new(AtomicUsize::new(0));
    let fence_reads = Arc::clone(&reads);
    let hosted = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), move || {
        assert!(door.turn.is_locked(), "the fence is read inside the turn");
        fence_reads.fetch_add(1, Ordering::Relaxed);
        Err(Status::failed_precondition("this replica does not lead"))
    });
    let refused = Err(MembershipWriteRefused::NotLeading(
        "this replica does not lead".to_owned(),
    ));
    assert_eq!(
        hosted.found_one(&maker, URL, &|_| panic!(
            "a follower does not ask for facts"
        )),
        refused
    );
    assert_eq!(hosted.drained(by, URL, seen), refused);
    assert_eq!(hosted.gone(by, URL, seen), refused);
    assert_eq!(reads.load(Ordering::Relaxed), 3);
    assert_eq!(store.by_key(URL), row);
    assert_eq!(store.epoch(), epoch);
    assert_eq!(hosted.door.place(), at);
}

/// A provider's write on the replica that leads is made in the door's
/// turn: the fence is read and the row's facts are stated while the turn
/// is held, and the call holds it no more once it answers.
#[test]
fn a_providers_write_on_the_replica_that_leads_is_made_in_the_doors_turn() {
    let (store, door) = door();
    let door = Arc::new(door);
    let fenced = Arc::clone(&door);
    let hosted = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), move || {
        assert!(fenced.turn.is_locked(), "the fence is read inside the turn");
        Ok(TERM)
    });
    let maker = found("pods/a");
    let stated = AtomicBool::new(false);
    hosted
        .found_one(&maker, URL, &|_| {
            assert!(
                door.turn.is_locked(),
                "the facts are stated inside the turn"
            );
            stated.store(true, Ordering::Relaxed);
            found_facts()
        })
        .expect("registered");
    assert!(
        stated.load(Ordering::Relaxed),
        "the write asked for the facts"
    );
    assert!(!door.turn.is_locked(), "the call let the turn go");
    assert!(store.by_key(URL).is_some(), "the row stands");
}

#[test]
fn the_two_doors_answer_one_row_alike() {
    let (replica, door) = door();
    let local = Arc::new(MemoryStore::default());
    let maker = found("pods/a");
    let hosted = ReplicaMembershipWriter::new(Arc::new(door), operator(), under(TERM));
    let lone = StoreDoor::new(Arc::clone(&local));
    let write = |door: &dyn Rows, store: &MemoryStore| {
        let by = Remover::Registrant(&maker);
        assert_eq!(
            door.gone(
                by,
                URL,
                RowRevision {
                    row: WorkerId(0),
                    revision: 0
                }
            ),
            Err(MembershipWriteRefused::Absent)
        );
        let registered = door
            .found_one(&maker, URL, &|_| found_facts())
            .expect("registered");
        let first = store.by_key(URL).expect("the first row");
        assert_eq!(registered, MembershipWriteOutcome::Registered(first.id));
        assert_eq!(
            door.found_one(&maker, URL, &|_| found_facts()),
            Ok(MembershipWriteOutcome::Unmoved(first.id))
        );
        let refresh = || WorkerFacts {
            max_running: Some(8),
            ..found_facts()
        };
        assert_eq!(
            door.found_one(&maker, URL, &|_| refresh()),
            Ok(MembershipWriteOutcome::InPlace(first.id))
        );
        let refreshed = store.by_key(URL).expect("the refreshed row");
        let wrote = door.found_one(&maker, URL, &|_| WorkerFacts {
            role: Role::Decode,
            ..refresh()
        });
        let fresh = store.by_key(URL).expect("the new row");
        assert_eq!(
            wrote,
            Ok(MembershipWriteOutcome::Another {
                old: first.id,
                new: fresh.id
            })
        );
        let seen = standing(&fresh);
        assert_eq!(
            door.drained(by, URL, seen),
            Ok(MembershipWriteOutcome::Draining(fresh.id))
        );
        let draining = store.by_key(URL).expect("the draining row");
        assert_eq!(
            door.drained(by, URL, seen),
            Ok(MembershipWriteOutcome::Unmoved(fresh.id))
        );
        assert_eq!(
            door.gone(by, URL, seen),
            Ok(MembershipWriteOutcome::Removed(fresh.id))
        );
        assert_eq!(store.by_key(URL), None);
        [first, refreshed, fresh, draining].map(|row| {
            assert_eq!(row.id, identity(URL, row.facts.incarnation));
            (
                WorkerFacts {
                    incarnation: 0,
                    ..row.facts
                },
                row.revision,
                row.status,
                row.maker,
            )
        })
    };
    assert_eq!(write(&lone, &local), write(&hosted, &replica));
    assert_eq!(local.epoch(), replica.epoch());
}

/// The controller's door reads a built worker's facts alone. Its log
/// carries no pairing descriptor and its room test refuses a row that
/// states one. So a worker that reports a transport and a version is
/// registered and refreshed here, and its row states no descriptor.
#[test]
fn a_worker_that_reports_pairing_facts_is_laid_with_no_descriptor() {
    const HTTP: &str = "http://10.0.0.6:8000";
    let (store, door) = hosted_door();
    let built = |version: &str| -> Arc<dyn Worker> {
        let worker = <BasicWorkerBuilder>::new(HTTP)
            .model(ModelCard::new("m"))
            .runtime_type(smg_worker::worker::RuntimeType::Sglang)
            .label("disaggregation_transfer_backend", "nixl");
        Arc::new(worker.label("version", version).build())
    };
    let first = built("1");
    let offered = first.pd_pairing();
    assert_eq!(
        (offered.transport(), offered.version()),
        (Some("nixl"), Some("1"))
    );
    let wrote = door.found_built(&Registrant::Control, Precondition::Absent, &first, None);
    assert!(wrote.is_ok(), "{wrote:?}");
    let row = store.by_key(HTTP).expect("the registered row");
    assert_eq!(row.facts.pairing, None);

    let wrote = door.refreshed(standing(&row), &built("2"), &first);
    assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)));
    let refreshed = store.by_key(HTTP).expect("the refreshed row");
    assert_eq!(refreshed.facts.pairing, None);
    let version = refreshed.facts.labels.get("version");
    assert_eq!(version.map(String::as_str), Some("2"));
}

/// A registration may state the default's own window in so many words:
/// its row keeps the number, and the object built from the row waits it.
/// A refresh reads that object back as its row states it, so it moves the
/// row's labels, as it does for a row that states another window or none.
#[test]
fn a_row_that_states_the_default_window_takes_a_label_refresh() {
    use smg_worker::worker::HealthCheckConfig;

    let default = u32::try_from(HealthCheckConfig::default().drain_settle_secs)
        .expect("the default window in a row's width");
    let (store, door) = door();
    let door = Arc::new(door);
    let hosted = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM));
    for (host, window) in [(1, None), (2, Some(default + 40)), (3, Some(default))] {
        let url = format!("grpc://10.0.1.{host}:8000");
        let stated = proto::WorkerSpec {
            drain_settle_secs: window,
            ..unchecked(&url)
        };
        door.register(&operator(), under(TERM), &ask(stated, Mode::Add))
            .expect("one row");
        let row = store.by_key(&url).expect("the registered row");
        assert_eq!(row.facts.drain_settle_secs, window);
        // What a refresh rebuilt: the row's object under another label.
        let first = crate::worker_host::object(&row);
        let relabelled = WorkerRow {
            facts: WorkerFacts {
                labels: [("tier".to_owned(), "silver".to_owned())].into(),
                ..row.facts.clone()
            },
            ..row.clone()
        };
        let built = crate::worker_host::object(&relabelled);
        let wrote = hosted.refreshed(standing(&row), &built, &first);
        assert_eq!(
            wrote,
            Ok(MembershipWriteOutcome::InPlace(row.id)),
            "a row that states {window:?}"
        );
        let refreshed = store.by_key(&url).expect("the refreshed row");
        assert_eq!(refreshed.facts.drain_settle_secs, window);
        let tier = refreshed.facts.labels.get("tier");
        assert_eq!(tier.map(String::as_str), Some("silver"));
    }
}

/// A registration may state a health policy with a part of a second, and a
/// runtime the worker crate does not know or writes otherwise. The row
/// keeps each as stated, and its object holds a narrower form: the object
/// does not read back as the row. A refresh holds its worker against the
/// row's own object, so the labels move and the row keeps its own word of
/// every other fact.
#[test]
fn a_row_that_states_what_its_object_holds_narrower_takes_a_label_refresh() {
    use smg_worker::worker::{HealthCheckConfig, RuntimeType};

    let default = u32::try_from(HealthCheckConfig::default().drain_settle_secs)
        .expect("the default window in a row's width");
    let (store, door) = door();
    let door = Arc::new(door);
    let hosted = ReplicaMembershipWriter::new(Arc::clone(&door), operator(), under(TERM));
    let policy = |timeout_ms: u32, check_interval_ms: u32| {
        Some(proto::HealthPolicy {
            disabled: true,
            timeout_ms,
            check_interval_ms,
            ..Default::default()
        })
    };
    let last = "grpc://10.0.2.6:8000";
    // Each registration, and the runtime its row's object holds.
    let rows: [(proto::WorkerSpec, RuntimeType); 6] = [
        (
            proto::WorkerSpec {
                health_policy: policy(1_500, 0),
                ..spec("grpc://10.0.2.1:8000")
            },
            RuntimeType::Unspecified,
        ),
        (
            proto::WorkerSpec {
                health_policy: policy(0, 2_500),
                ..spec("grpc://10.0.2.2:8000")
            },
            RuntimeType::Unspecified,
        ),
        (
            proto::WorkerSpec {
                runtime: "custom-engine".to_owned(),
                ..unchecked("grpc://10.0.2.3:8000")
            },
            RuntimeType::Unspecified,
        ),
        (
            proto::WorkerSpec {
                runtime: "VLLM".to_owned(),
                ..unchecked("grpc://10.0.2.4:8000")
            },
            RuntimeType::Vllm,
        ),
        // The worker crate's own word for no runtime.
        (
            proto::WorkerSpec {
                runtime: "unspecified".to_owned(),
                ..unchecked("grpc://10.0.2.5:8000")
            },
            RuntimeType::Unspecified,
        ),
        // Each such fact at once, beside a window at the default's number.
        (
            proto::WorkerSpec {
                runtime: "TensorRT-LLM".to_owned(),
                health_policy: policy(1_500, 2_500),
                drain_settle_secs: Some(default),
                ..spec(last)
            },
            RuntimeType::Trtllm,
        ),
    ];
    for (stated, runtime) in rows {
        let url = stated.url.clone();
        let named = stated.runtime.clone();
        door.register(&operator(), under(TERM), &ask(stated, Mode::Add))
            .expect("one row");
        let row = store.by_key(&url).expect("the registered row");
        assert_eq!(row.facts.engine_kind, named, "{url}");
        // The row's object holds the runtime the worker crate reads the
        // name as, and does not read back as the row.
        let first = crate::worker_host::object(&row);
        assert_eq!(first.metadata().spec.runtime_type, runtime, "{url}");
        let read = crate::worker_host::facts(first.as_ref(), Some(&row.facts));
        assert_ne!(read, row.facts, "{url}");
        // What a refresh rebuilt: the row's object under another label.
        let relabelled = WorkerRow {
            facts: WorkerFacts {
                labels: [("tier".to_owned(), "silver".to_owned())].into(),
                ..row.facts.clone()
            },
            ..row.clone()
        };
        let built = crate::worker_host::object(&relabelled);
        let wrote = hosted.refreshed(standing(&row), &built, &first);
        assert_eq!(wrote, Ok(MembershipWriteOutcome::InPlace(row.id)), "{url}");
        let refreshed = store.by_key(&url).expect("the refreshed row");
        assert_eq!(refreshed.facts, relabelled.facts, "{url}");
    }
    let kept = store.by_key(last).expect("the last row").facts;
    assert_eq!(
        (
            kept.engine_kind.as_str(),
            kept.health.timeout_ms,
            kept.health.check_interval_ms,
            kept.drain_settle_secs
        ),
        ("TensorRT-LLM", 1_500, 2_500, Some(default))
    );
}
