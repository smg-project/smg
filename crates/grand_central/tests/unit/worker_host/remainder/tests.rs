use std::{collections::BTreeMap, sync::Arc};

use smg_worker::{
    health::overload::OverloadThresholds, registry::WorkerRegistry, worker::HealthCheckConfig,
};

use super::RowRemainder;
use crate::{
    engine::{
        demands::DemandFacts,
        view::{Role, WireModes, WorkerId},
    },
    store::{
        encoding::{EncodeHandoff, EncodeOffer},
        pairing::PairingDescriptor,
        GatewayId, HealthPolicy, MemoryStore, Overload, Reachability, Registrant, Signals,
        WorkerFacts, WorkerPatch, WorkerRow, WorkerStatus,
    },
    worker_host::{objects::tests::stand, Members, Objects},
};

fn holder() -> Objects<Members> {
    Objects::new(Arc::new(WorkerRegistry::new()))
}

fn at(key: &str) -> WorkerFacts {
    WorkerFacts {
        key: key.to_owned(),
        ..WorkerFacts::default()
    }
}

/// The row `read` against the row that `stood`: each fact under its
/// name, then the two whole.
fn same(read: &WorkerRow, stood: &WorkerRow, case: &str) {
    macro_rules! fact {
            ($($name:ident),+) => {
                $(assert_eq!(
                    read.facts.$name,
                    stood.facts.$name,
                    "{case}: {}",
                    stringify!($name)
                );)+
            };
        }
    fact!(
        key,
        engine,
        incarnation,
        role,
        wire_modes,
        engine_kind,
        models,
        dp_rank,
        dp_size,
        labels,
        zone,
        pairing,
        encode,
        vision,
        max_running,
        kv_block_size,
        bootstrap_port,
        priority,
        reachable_from,
        signals,
        health,
        overload,
        drain_settle_secs,
        demand_facts
    );
    assert_eq!(read.id, stood.id, "{case}: id");
    assert_eq!(read.status, stood.status, "{case}: status");
    assert_eq!(read.revision, stood.revision, "{case}: revision");
    assert_eq!(read.admitted_at, stood.admitted_at, "{case}: admitted_at");
    assert_eq!(read.maker, stood.maker, "{case}: maker");
    assert_eq!(read, stood, "{case}");
}

/// Write the row `facts` state through the holder: the row's id.
fn laid(objects: &Objects<Members>, store: &MemoryStore, facts: WorkerFacts) -> WorkerId {
    let written = objects.upsert(store, facts, Registrant::Control);
    written.expect("the store takes the write").id()
}

/// Hold every row of `store` against the row its object and its
/// remainder state. Return how many rows were held.
fn read_back(store: &MemoryStore, objects: &Objects<Members>, case: &str) -> usize {
    let (_, rows) = store.all_rows();
    for stood in &rows {
        let object = objects.object(stood.id).expect("the row's object");
        let read = objects.row_of(&object).expect("the row read back");
        same(&read, stood, &format!("{case}, {}", stood.facts.key));
        let beside = objects.remainder(&object).expect("the remainder");
        assert_eq!(*beside, RowRemainder::of(stood), "{case}");
    }
    rows.len()
}

/// A row for each form a fact takes that its object holds narrower or
/// not at all, each under an address of its own.
fn fleet() -> Vec<WorkerFacts> {
    let wires = [
        ("http://http:8000", WireModes::HTTP),
        ("grpc://grpc:8000", WireModes::GRPC),
        ("zmq://local:8000", WireModes::ZMQ),
        ("http://two:8000", WireModes::HTTP.union(WireModes::GRPC)),
        ("http://near:8000", WireModes::HTTP.union(WireModes::ZMQ)),
        (
            "http://three:8000",
            WireModes::HTTP.union(WireModes::GRPC).union(WireModes::ZMQ),
        ),
    ];
    let runtimes = [
        ("http://known:8000", "generic"),
        ("http://written:8000", "GENERIC"),
        ("http://unknown:8000", "no-such-runtime"),
        ("http://unsaid:8000", "unspecified"),
    ];
    let policies = [
        ("http://parts:8000", 1_500, 2_001, true),
        ("http://none:8000", 0, 0, true),
        ("http://off:8000", 7_000, 2_000, false),
    ];
    let windows = [
        ("http://window:8000", 30),
        (
            "http://default:8000",
            u32::try_from(HealthCheckConfig::default().drain_settle_secs).expect("a window"),
        ),
    ];
    let thresholds = [
        ("http://both:8000", Some(800), Some(12)),
        ("http://one:8000", None, Some(7)),
    ];
    let roles = [
        ("http://prefill:8000", Role::Prefill),
        ("http://decode:8000", Role::Decode),
        ("http://encode:8000", Role::Encode),
    ];
    let mut fleet = vec![
        // What a registration leaves unsaid: no engine, no wire, no
        // width, no model.
        at("http://bare:8000"),
        // Its own engine, named.
        WorkerFacts {
            engine: "http://own:8000".to_owned(),
            dp_size: 1,
            ..at("http://own:8000")
        },
        // A rank of an engine, with every fact the object holds whole.
        WorkerFacts {
            engine: "grpc://engine:9000".to_owned(),
            models: vec!["m".to_owned(), "n".to_owned()],
            dp_rank: 1,
            dp_size: 2,
            labels: BTreeMap::from([
                ("tier".to_owned(), "gold".to_owned()),
                ("zone".to_owned(), "east".to_owned()),
            ]),
            kv_block_size: Some(32),
            bootstrap_port: Some(8_998),
            ..at("grpc://engine:9000@1")
        },
        // The engine whole: the object holds its width and no rank.
        WorkerFacts {
            engine: "http://whole:8000".to_owned(),
            dp_size: 4,
            ..at("http://whole:8000")
        },
        // A rank beside a width of one: the object holds neither.
        WorkerFacts {
            dp_rank: 3,
            dp_size: 1,
            ..at("http://rank:8000")
        },
        // Every fact no field of the object holds.
        WorkerFacts {
            incarnation: 7,
            zone: Some("east".to_owned()),
            pairing: Some(Arc::new(PairingDescriptor {
                protocol: Some("pd".to_owned()),
                ..PairingDescriptor::default()
            })),
            encode: EncodeOffer {
                handoff: EncodeHandoff::Push,
                producer: true,
                ..EncodeOffer::default()
            },
            vision: true,
            max_running: Some(23),
            priority: -4,
            reachable_from: Reachability::Gateway(GatewayId(42)),
            signals: Signals {
                events: true,
                loads: true,
                scrape_interval_ms: 3_000,
            },
            demand_facts: DemandFacts {
                runtime: 7,
                window: 8_192,
                ..DemandFacts::default()
            },
            ..at("http://beside:8000")
        },
    ];
    fleet.extend(wires.map(|(key, wire_modes)| WorkerFacts {
        wire_modes,
        ..at(key)
    }));
    fleet.extend(runtimes.map(|(key, name)| WorkerFacts {
        engine_kind: name.to_owned(),
        ..at(key)
    }));
    fleet.extend(
        policies.map(|(key, check_interval_ms, timeout_ms, probes)| WorkerFacts {
            health: HealthPolicy {
                failure_threshold: 4,
                success_threshold: 5,
                check_interval_ms,
                timeout_ms,
                probes,
            },
            ..at(key)
        }),
    );
    fleet.extend(windows.map(|(key, window)| WorkerFacts {
        drain_settle_secs: Some(window),
        ..at(key)
    }));
    fleet.extend(
        thresholds.map(|(key, kv_usage_permille, waiting_requests)| WorkerFacts {
            overload: Overload {
                kv_usage_permille,
                waiting_requests,
            },
            ..at(key)
        }),
    );
    fleet.extend(roles.map(|(key, role)| WorkerFacts { role, ..at(key) }));
    fleet
}

/// Every row reads back from its object and what the holder keeps
/// beside it, fact by fact: under a holder that asks nothing, under one
/// that builds a host-local row with its checks off, as a controller's
/// does, and under one whose objects state thresholds. No table of rows
/// is read for it.
#[test]
fn a_row_reads_back_from_its_object_and_what_stands_beside_it() {
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: Some(0.8),
    };
    let holders = [
        ("a plain holder", holder()),
        ("checks off", holder().with_host_local_unchecked()),
        ("thresholds", holder().with_overload_thresholds(defaults)),
    ];
    for (case, objects) in holders {
        let store = MemoryStore::new();
        let fleet = fleet();
        let rows = fleet.len();
        for facts in fleet {
            laid(&objects, &store, facts);
        }
        assert_eq!(read_back(&store, &objects, case), rows);
    }
}

/// A row's maker, its status and its later revisions read back too: a
/// patch moves the revision and the facts it names, another incarnation
/// is another row, and a row that left reads back nothing.
#[test]
fn a_rows_maker_its_status_and_its_later_lives_read_back() {
    const URL: &str = "http://w:8000";
    let store = MemoryStore::new();
    let objects = holder();
    let maker = Registrant::provider("found", "one").expect("a maker");
    let written = objects.upsert(&store, at(URL), maker);
    let id = written.expect("the store takes the write").id();
    laid(&objects, &store, at("http://control:8000"));
    assert_eq!(read_back(&store, &objects, "two makers"), 2);

    // A status stands on the object, and the store is told it.
    for status in [
        WorkerStatus::Ready,
        WorkerStatus::NotReady,
        WorkerStatus::Failed,
        WorkerStatus::Draining,
    ] {
        stand(&objects, &store, id, status);
        read_back(&store, &objects, &format!("{status:?}"));
    }

    let revision = store.get(id).expect("the row").revision;
    let patch = WorkerPatch {
        labels: Some(BTreeMap::from([("tier".to_owned(), "gold".to_owned())])),
        priority: Some(3),
        overload: Some(Overload {
            kv_usage_permille: Some(650),
            waiting_requests: None,
        }),
    };
    objects.patch(&store, id, patch, revision).expect("a patch");
    read_back(&store, &objects, "a patch");
    let object = objects.object(id).expect("the row's object");
    let first = objects.remainder(&object).expect("the remainder");
    let again = objects.remainder(&object).expect("the remainder");
    assert!(Arc::ptr_eq(&first, &again), "one remainder a row laid");
    assert!(first.standing.revision > revision);

    let next = WorkerFacts {
        incarnation: 2,
        vision: true,
        ..at(URL)
    };
    let next = laid(&objects, &store, next);
    assert_ne!(next, id);
    assert_eq!(read_back(&store, &objects, "another life"), 2);
    assert_eq!(objects.object(id), None);

    let object = objects.object(next).expect("the row's object");
    let revision = store.get(next).expect("the row").revision;
    objects.remove(&store, next, revision).expect("a removal");
    assert_eq!(read_back(&store, &objects, "a removal"), 1);
    assert_eq!(objects.row_of(&object), None);
    assert_eq!(objects.remainder(&object), None);
}

/// Another history of the store may hold a row under a held id and
/// revision, with equal facts, that another made: a whole read keeps
/// the object and takes the remainder again.
#[test]
fn a_row_another_history_made_reads_back_its_maker_under_the_kept_object() {
    const URL: &str = "http://w:8000";
    let store = MemoryStore::new();
    let registry = Arc::new(WorkerRegistry::new());
    let objects = Objects::new(Arc::clone(&registry));
    let first = Registrant::provider("found", "one").expect("a maker");
    let written = objects.upsert(&store, at(URL), first);
    let id = written.expect("the store takes the write").id();
    assert_eq!(read_back(&store, &objects, "the first history"), 1);
    let object = objects.object(id).expect("the row's object");
    let built = registry.get(&object).expect("the object");
    let before = objects.remainder(&object).expect("the remainder");

    let other = MemoryStore::new();
    let second = Registrant::provider("found", "two").expect("a maker");
    assert_eq!(other.upsert_by(at(URL), second.clone()).id(), id);
    assert!(objects.restore_from(&store, other.checkpoint()));
    assert_eq!(read_back(&store, &objects, "another history"), 1);

    let after = objects.remainder(&object).expect("the remainder");
    assert_eq!(after.standing, before.standing);
    assert_eq!(after.maker, second);
    let kept = registry.get(&object).expect("the object");
    assert!(Arc::ptr_eq(&built, &kept), "the object is kept");
}

/// A rank's row that states every fact.
fn stated() -> WorkerFacts {
    WorkerFacts {
        key: "grpc://engine:9000@1".to_owned(),
        engine: "grpc://engine:9000".to_owned(),
        incarnation: 3,
        role: Role::Prefill,
        wire_modes: WireModes::GRPC,
        engine_kind: "generic".to_owned(),
        models: vec!["m".to_owned(), "n".to_owned()],
        dp_rank: 1,
        dp_size: 4,
        labels: BTreeMap::from([("tier".to_owned(), "gold".to_owned())]),
        zone: Some("east".to_owned()),
        pairing: Some(Arc::new(PairingDescriptor {
            protocol: Some("pd".to_owned()),
            ..PairingDescriptor::default()
        })),
        encode: EncodeOffer {
            handoff: EncodeHandoff::Push,
            producer: true,
            ..EncodeOffer::default()
        },
        vision: true,
        max_running: Some(23),
        kv_block_size: Some(32),
        bootstrap_port: Some(8_998),
        priority: -4,
        reachable_from: Reachability::Gateway(GatewayId(42)),
        signals: Signals {
            events: true,
            loads: true,
            scrape_interval_ms: 3_000,
        },
        health: HealthPolicy {
            failure_threshold: 4,
            success_threshold: 5,
            check_interval_ms: 1_500,
            timeout_ms: 2_001,
            probes: true,
        },
        overload: Overload {
            kv_usage_permille: Some(800),
            waiting_requests: Some(12),
        },
        drain_settle_secs: Some(30),
        demand_facts: DemandFacts {
            runtime: 7,
            window: 8_192,
            ..DemandFacts::default()
        },
    }
}

/// One move of a row's fact, under the fact's name.
type Move = (&'static str, fn(&mut WorkerFacts));

/// One move of each fact a row can change within one life: every fact
/// but the key and the incarnation. Each moves [`stated`]'s row, and the
/// row a bare worker is found with.
fn moves() -> [Move; 22] {
    [
        ("engine", |facts: &mut WorkerFacts| {
            facts.engine = "grpc://moved:9000".to_owned();
        }),
        ("role", |facts: &mut WorkerFacts| facts.role = Role::Decode),
        ("wire_modes", |facts: &mut WorkerFacts| {
            facts.wire_modes = WireModes::HTTP.union(WireModes::ZMQ);
        }),
        ("engine_kind", |facts: &mut WorkerFacts| {
            facts.engine_kind = "no-such-runtime".to_owned();
        }),
        ("models", |facts: &mut WorkerFacts| {
            facts.models = vec!["moved".to_owned()];
        }),
        ("dp_rank", |facts: &mut WorkerFacts| facts.dp_rank = 2),
        ("dp_size", |facts: &mut WorkerFacts| facts.dp_size = 8),
        ("labels", |facts: &mut WorkerFacts| {
            facts.labels = BTreeMap::from([("tier".to_owned(), "moved".to_owned())]);
        }),
        ("zone", |facts: &mut WorkerFacts| {
            facts.zone = Some("west".to_owned());
        }),
        ("pairing", |facts: &mut WorkerFacts| {
            facts.pairing = Some(Arc::new(PairingDescriptor {
                protocol: Some("moved".to_owned()),
                ..PairingDescriptor::default()
            }));
        }),
        ("encode", |facts: &mut WorkerFacts| {
            facts.encode = EncodeOffer {
                producer: !facts.encode.producer,
                ..EncodeOffer::default()
            };
        }),
        ("vision", |facts: &mut WorkerFacts| {
            facts.vision = !facts.vision;
        }),
        ("max_running", |facts: &mut WorkerFacts| {
            facts.max_running = Some(99);
        }),
        ("kv_block_size", |facts: &mut WorkerFacts| {
            facts.kv_block_size = Some(64);
        }),
        ("bootstrap_port", |facts: &mut WorkerFacts| {
            facts.bootstrap_port = Some(9_001);
        }),
        ("priority", |facts: &mut WorkerFacts| facts.priority = 9),
        ("reachable_from", |facts: &mut WorkerFacts| {
            facts.reachable_from = Reachability::Gateway(GatewayId(7));
        }),
        ("signals", |facts: &mut WorkerFacts| {
            facts.signals = Signals {
                events: true,
                loads: false,
                scrape_interval_ms: 1_000,
            };
        }),
        ("health", |facts: &mut WorkerFacts| {
            facts.health = HealthPolicy {
                failure_threshold: 9,
                success_threshold: 9,
                check_interval_ms: 9_500,
                timeout_ms: 9_001,
                probes: true,
            };
        }),
        ("overload", |facts: &mut WorkerFacts| {
            facts.overload = Overload {
                kv_usage_permille: Some(999),
                waiting_requests: Some(99),
            };
        }),
        ("drain_settle_secs", |facts: &mut WorkerFacts| {
            facts.drain_settle_secs = Some(99);
        }),
        ("demand_facts", |facts: &mut WorkerFacts| {
            facts.demand_facts = DemandFacts {
                runtime: 9,
                window: 4_096,
                ..DemandFacts::default()
            };
        }),
    ]
}

/// Each fact of a row changed alone reads back as it was changed, under
/// a holder that builds its objects, as a controller's does: a fact the
/// object holds comes back off the object built again in the standing
/// one's place, under its id. Another key is another row beside the
/// first, and another incarnation is another row in its place.
#[test]
fn each_fact_changed_alone_reads_back_as_it_was_changed() {
    for (fact, change) in moves() {
        let store = MemoryStore::new();
        let objects = holder().with_host_local_unchecked();
        let id = laid(&objects, &store, stated());
        assert_eq!(read_back(&store, &objects, fact), 1);
        let object = objects.object(id).expect("the row's object");

        let mut facts = stated();
        change(&mut facts);
        assert_ne!(facts, stated(), "{fact}: the fact moved");
        assert_eq!(laid(&objects, &store, facts), id, "{fact}: one life");
        assert_eq!(read_back(&store, &objects, fact), 1);
        let again = objects.object(id);
        assert_eq!(again.as_ref(), Some(&object), "{fact}: the object's id");
    }

    let store = MemoryStore::new();
    let objects = holder().with_host_local_unchecked();
    let id = laid(&objects, &store, stated());
    let beside = WorkerFacts {
        key: "grpc://engine:9000@2".to_owned(),
        ..stated()
    };
    assert_ne!(laid(&objects, &store, beside), id);
    assert_eq!(read_back(&store, &objects, "key"), 2);
    let next = WorkerFacts {
        incarnation: 4,
        ..stated()
    };
    assert_ne!(laid(&objects, &store, next), id);
    assert_eq!(read_back(&store, &objects, "incarnation"), 2);
    assert_eq!(objects.object(id), None);
}

/// A cut laid whole holds each row against the row read back off its
/// object. A row that did not read back as stated would be laid again at
/// every cut. So a second cut keeps the very object of each row of a
/// varied fleet. The fleet is [`fleet`], the row that states every fact
/// and a row for each fact moved alone. It also holds a key that ends in
/// a slash, and blank labels under the two names a worker's pairing
/// facts are read from.
///
/// Each holds under a plain holder, under one that builds a host-local
/// row with its checks off, and under one whose objects state thresholds.
#[test]
fn a_second_cut_keeps_the_very_object_of_each_row_of_a_varied_fleet() {
    let defaults = OverloadThresholds {
        waiting_requests: Some(8),
        token_usage: Some(0.8),
    };
    let holders = [
        ("a plain holder", holder()),
        ("checks off", holder().with_host_local_unchecked()),
        ("thresholds", holder().with_overload_thresholds(defaults)),
    ];
    for (case, objects) in holders {
        let store = MemoryStore::new();
        let mut fleet = fleet();
        fleet.push(WorkerFacts {
            key: "grpc://engine:9000@whole".to_owned(),
            ..stated()
        });
        for (n, (_, change)) in moves().into_iter().enumerate() {
            let mut facts = WorkerFacts {
                key: format!("grpc://engine:9000@moved{n}"),
                ..stated()
            };
            change(&mut facts);
            fleet.push(facts);
        }
        fleet.push(at("http://slash:8000/"));
        fleet.push(WorkerFacts {
            labels: BTreeMap::from([
                ("pairing_protocol".to_owned(), " ".to_owned()),
                ("kv_connector".to_owned(), String::new()),
            ]),
            ..at("http://blank:8000")
        });
        let rows = fleet.len();
        for facts in fleet {
            laid(&objects, &store, facts);
        }
        let registry = objects.registry();
        let object = |row: &WorkerRow| {
            let id = objects.object(row.id).expect("the row's object");
            registry.get(&id).expect("the object")
        };
        let (_, table) = store.all_rows();
        assert_eq!(table.len(), rows, "{case}: one row a key");
        let built: Vec<_> = table.iter().map(object).collect();
        for cut in ["a first cut", "a second cut"] {
            assert!(objects.restore_from(&store, store.checkpoint()), "{case}");
            for (row, built) in table.iter().zip(&built) {
                let key = &row.facts.key;
                assert!(Arc::ptr_eq(built, &object(row)), "{case}, {cut}: {key}");
            }
            assert_eq!(read_back(&store, &objects, case), rows);
        }
    }
}
