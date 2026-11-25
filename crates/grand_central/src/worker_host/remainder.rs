//! What of a row its object does not hold as stated: the holder keeps it
//! beside the object, and the two together state the row.

use std::{sync::Arc, time::Instant};

use smg_worker::identity::basic::Worker;

use super::{found::facts, object::row_status, objects::RowRevision};
use crate::{
    engine::{demands::DemandFacts, view::WireModes},
    store::{
        encoding::EncodeOffer, pairing::PairingDescriptor, HealthPolicy, Overload, Reachability,
        Registrant, Signals, WorkerFacts, WorkerRow,
    },
};

/// What a worker's row states and its object does not hold as stated. The
/// holder keeps one beside each object, and the two together state the row
/// ([`Objects::row_of`](super::Objects::row_of)).
///
/// Of the 24 facts a row states it holds 17 as stated and one bit of an
/// eighteenth, the engine, and nothing the object holds whole: the key,
/// the role, the models, the labels, the cache block size and the
/// bootstrap port are the object's, and so is the status.
///
/// Eight facts stand twice, in a narrower form on the object and stated
/// here; of the engine one bit stands here. The rank and the width share
/// a line:
///
/// | The fact | The object's form | What the object cannot say |
/// |---|---|---|
/// | `engine` | the address it dials, where that is not its key | whether the row named its key as its engine or named none: one bit here |
/// | `dp_rank`, `dp_size` | a width above one, and a rank unless the row is its engine whole | a width of one or of none, and a rank stated beside either |
/// | `wire_modes` | the one wire it speaks | the other wires of the set |
/// | `engine_kind` | the runtime the worker crate reads the name as | a name that crate does not know, and a known name as it was written |
/// | `health` | its checks in whole seconds, off where its host cannot dial the worker | a part of a second, and the policy of a row built with its checks off |
/// | `overload` | the thresholds its host's monitor judges by, the row's own over the host's defaults, an external row's own alone, or none | which of them the row stated |
/// | `drain_settle_secs` | the window in force | a window stated equal to the object's default, which reads back as none |
///
/// Ten are not carried to the object: the incarnation, and the facts a
/// decision or a collector reads of the row alone. Three of them have a
/// place of the object's own that no row fills: its priority is unsigned,
/// it derives its pairing facts at its build, and it reads its admission
/// window off a label. The row's id and revision, who made it and when
/// stand here too.
///
/// It stays inside the crate: its readers are the crate's own, and a fact
/// leaves it when the object gets a field that holds the fact as stated.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RowRemainder {
    /// The row's id, and the revision this was taken at.
    pub standing: RowRevision,
    /// Who made the row.
    pub maker: Registrant,
    /// When the row was created on the host's clock.
    pub admitted_at: Instant,
    /// The store's number for this life of the key.
    pub incarnation: u64,
    /// Whether the row names an engine. One that names none dials its key,
    /// as one that names its key does.
    pub names_engine: bool,
    /// The rank as stated.
    pub dp_rank: u16,
    /// The width as stated.
    pub dp_size: u16,
    /// The whole set of wires.
    pub wire_modes: WireModes,
    /// The engine family as the handshake named it.
    pub engine_kind: String,
    /// The policy in its own milliseconds.
    pub health: HealthPolicy,
    /// The thresholds the registration names for itself, and no default.
    pub overload: Overload,
    /// The drain window, where the registration states one.
    pub drain_settle_secs: Option<u32>,
    /// The zone.
    pub zone: Option<String>,
    /// What the worker offers a prefill-decode partner.
    pub pairing: Option<Arc<PairingDescriptor>>,
    /// The worker's end of the encode handoff.
    pub encode: EncodeOffer,
    /// Whether the worker takes images.
    pub vision: bool,
    /// The engine's admission window.
    pub max_running: Option<u32>,
    /// The priority, signed as the row states it.
    pub priority: i32,
    /// From where the worker is reachable.
    pub reachable_from: Reachability,
    /// How the worker's signals reach the store.
    pub signals: Signals,
    /// What the worker states for a request's hard demands.
    pub demand_facts: DemandFacts,
}

impl RowRemainder {
    /// What `row` states and its object does not hold as stated. The row is
    /// taken apart with no rest pattern, so a fact added to the row does
    /// not build until this function says where it stands.
    pub(super) fn of(row: &WorkerRow) -> Self {
        let WorkerRow {
            id,
            facts,
            status: _,
            revision,
            admitted_at,
            maker,
        } = row;
        let WorkerFacts {
            key: _,
            engine,
            incarnation,
            role: _,
            wire_modes,
            engine_kind,
            models: _,
            dp_rank,
            dp_size,
            labels: _,
            zone,
            pairing,
            encode,
            vision,
            max_running,
            kv_block_size: _,
            bootstrap_port: _,
            priority,
            reachable_from,
            signals,
            health,
            overload,
            drain_settle_secs,
            demand_facts,
        } = facts;
        Self {
            standing: RowRevision {
                row: *id,
                revision: *revision,
            },
            maker: maker.clone(),
            admitted_at: *admitted_at,
            incarnation: *incarnation,
            names_engine: !engine.is_empty(),
            dp_rank: *dp_rank,
            dp_size: *dp_size,
            wire_modes: *wire_modes,
            engine_kind: engine_kind.clone(),
            health: *health,
            overload: *overload,
            drain_settle_secs: *drain_settle_secs,
            zone: zone.clone(),
            pairing: pairing.clone(),
            encode: encode.clone(),
            vision: *vision,
            max_running: *max_running,
            priority: *priority,
            reachable_from: *reachable_from,
            signals: *signals,
            demand_facts: *demand_facts,
        }
    }

    /// The row `worker` and this remainder state together: the six facts
    /// the object holds whole as [`facts`] reads them of it now, its
    /// status, the address it dials as the engine of a row that names one,
    /// and every other fact as it was stated.
    ///
    /// It is the row the object was laid from, where nothing was set on
    /// the object since: models set on the object read back as set, under
    /// the revision this was taken at.
    pub(super) fn row(&self, worker: &dyn Worker) -> WorkerRow {
        let Self {
            standing,
            maker,
            admitted_at,
            incarnation,
            names_engine,
            dp_rank,
            dp_size,
            wire_modes,
            engine_kind,
            health,
            overload,
            drain_settle_secs,
            zone,
            pairing,
            encode,
            vision,
            max_running,
            priority,
            reachable_from,
            signals,
            demand_facts,
        } = self;
        let held = facts(worker, None);
        WorkerRow {
            id: standing.row,
            facts: WorkerFacts {
                key: held.key,
                // The object's address is the row's engine, where it named
                // one.
                engine: if *names_engine {
                    held.engine
                } else {
                    String::new()
                },
                incarnation: *incarnation,
                role: held.role,
                wire_modes: *wire_modes,
                engine_kind: engine_kind.clone(),
                models: held.models,
                dp_rank: *dp_rank,
                dp_size: *dp_size,
                labels: held.labels,
                zone: zone.clone(),
                pairing: pairing.clone(),
                encode: encode.clone(),
                vision: *vision,
                max_running: *max_running,
                kv_block_size: held.kv_block_size,
                bootstrap_port: held.bootstrap_port,
                priority: *priority,
                reachable_from: *reachable_from,
                signals: *signals,
                health: *health,
                overload: *overload,
                drain_settle_secs: *drain_settle_secs,
                demand_facts: *demand_facts,
            },
            status: row_status(worker.status()),
            revision: standing.revision,
            admitted_at: *admitted_at,
            maker: maker.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc};

    use smg_worker::{
        health::overload::OverloadThresholds, registry::WorkerRegistry, worker::HealthCheckConfig,
    };

    use super::RowRemainder;
    use crate::{
        engine::{
            demands::DemandFacts,
            view::{Role, WireModes},
        },
        store::{
            encoding::{EncodeHandoff, EncodeOffer},
            pairing::PairingDescriptor,
            GatewayId, HealthPolicy, MemoryStore, Overload, Reachability, Registrant, Signals,
            WorkerFacts, WorkerPatch, WorkerRow, WorkerStatus, WorkersStore as _,
        },
        worker_host::{Members, Objects},
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

    /// Follow `store`, then hold every row of it against the row its object
    /// and its remainder state. Return how many rows were held.
    fn read_back(store: &MemoryStore, objects: &Objects<Members>, case: &str) -> usize {
        objects.follow(store);
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
                store.upsert(facts);
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
        let objects = holder().with_status_followed();
        let maker = Registrant::provider("found", "one").expect("a maker");
        let id = store.upsert_by(at(URL), maker).id();
        store.upsert(at("http://control:8000"));
        assert_eq!(read_back(&store, &objects, "two makers"), 2);

        for status in [
            WorkerStatus::Ready,
            WorkerStatus::NotReady,
            WorkerStatus::Failed,
            WorkerStatus::Draining,
        ] {
            let revision = store.get(id).expect("the row").revision;
            store.set_status(id, status, revision).expect("a move");
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
        store.patch(id, patch, revision).expect("a patch");
        read_back(&store, &objects, "a patch");
        let object = objects.object(id).expect("the row's object");
        let first = objects.remainder(&object).expect("the remainder");
        let again = objects.remainder(&object).expect("the remainder");
        assert!(Arc::ptr_eq(&first, &again), "one remainder a row laid");
        assert!(first.standing.revision > revision);

        let next = store
            .upsert(WorkerFacts {
                incarnation: 2,
                vision: true,
                ..at(URL)
            })
            .id();
        assert_ne!(next, id);
        assert_eq!(read_back(&store, &objects, "another life"), 2);
        assert_eq!(objects.object(id), None);

        let object = objects.object(next).expect("the row's object");
        let revision = store.get(next).expect("the row").revision;
        store.remove(next, revision).expect("a removal");
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
        let id = store.upsert_by(at(URL), first).id();
        assert_eq!(read_back(&store, &objects, "the first history"), 1);
        let object = objects.object(id).expect("the row's object");
        let built = registry.get(&object).expect("the object");
        let before = objects.remainder(&object).expect("the remainder");

        let other = MemoryStore::new();
        let second = Registrant::provider("found", "two").expect("a maker");
        assert_eq!(other.upsert_by(at(URL), second.clone()).id(), id);
        assert!(store.restore_from(other.checkpoint()));
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
            let id = store.upsert(stated()).id();
            assert_eq!(read_back(&store, &objects, fact), 1);
            let object = objects.object(id).expect("the row's object");

            let mut facts = stated();
            change(&mut facts);
            assert_ne!(facts, stated(), "{fact}: the fact moved");
            assert_eq!(store.upsert(facts).id(), id, "{fact}: one life");
            assert_eq!(read_back(&store, &objects, fact), 1);
            let again = objects.object(id);
            assert_eq!(again.as_ref(), Some(&object), "{fact}: the object's id");
        }

        let store = MemoryStore::new();
        let objects = holder().with_host_local_unchecked();
        let id = store.upsert(stated()).id();
        let beside = WorkerFacts {
            key: "grpc://engine:9000@2".to_owned(),
            ..stated()
        };
        assert_ne!(store.upsert(beside).id(), id);
        assert_eq!(read_back(&store, &objects, "key"), 2);
        let next = WorkerFacts {
            incarnation: 4,
            ..stated()
        };
        assert_ne!(store.upsert(next).id(), id);
        assert_eq!(read_back(&store, &objects, "incarnation"), 2);
        assert_eq!(objects.object(id), None);
    }
}
