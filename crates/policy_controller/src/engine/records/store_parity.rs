//! The store's view of a fleet beside the record writer's.
//!
//! Two bodies of code hand a decision the records of a pool. The store
//! builds its view of the pool from its own rows
//! ([`StoreSnapshot::view`]), with the demand column beside it
//! ([`StoreSnapshot::demand_facts`]). The record writer builds
//! [`PoolRecords`] from worker objects. This case writes one fleet to a
//! store, has the worker host lay the fleet's objects
//! ([`Objects::follow`]), and compares what a decision is handed by each,
//! field by field. The store's count of the workers that stand outside the
//! document's lists ([`OutsideLists`]) is compared beside them: it is what
//! tells an operator that a zone was mistyped.
//!
//! How to read it:
//!
//! - [`fleet`] is the rows. Between them they state every fact a
//!   registration can state, but the ones [`NOT_STATED`] lists.
//! - [`DIFFERS`] lists the fields that differ between the two sides and are
//!   meant to agree: a line goes in the change that makes its field agree.
//! - [`BY_DESIGN`] lists the fields that differ because each side numbers
//!   what it holds in its own way.
//! - [`AT_REST`] lists the fields the fleet leaves at rest on both sides:
//!   the case says nothing of them yet.
//! - Every other field is set on some row, and agrees on every row.
//!
//! Every field of [`Candidate`], [`Health`], [`LoadRow`], [`PoolFacts`],
//! [`DemandFacts`], [`OutsideLists`] and [`WorkerFacts`] is taken apart
//! here with no rest pattern: a field added to one of them does not build
//! until this case names it. The case fails where a field differs and
//! stands on no list, and where a listed field differs no longer or rests
//! no longer: the change that moved the field strikes its line.
//!
//! [`StoreSnapshot::view`]: crate::store::StoreSnapshot::view
//! [`StoreSnapshot::demand_facts`]: crate::store::StoreSnapshot::demand_facts

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::{Duration, Instant},
};

use smg_worker::registry::WorkerRegistry;

use super::{DecisionClock, Live, PoolRecords};
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        demands::DemandFacts,
        view::{Candidate, Health, LoadRow, PoolFacts, Role, WireModes},
    },
    store::{
        pairing::PairingDescriptor, EncodeOffer, GatewayId, HealthPolicy, MemoryStore,
        OutsideLists, Overload, Reachability, Signals, ViewBuffers, ViewQuery, WorkerFacts,
        WorkerStatus, WorkersStore as _,
    },
    worker_host::{Members, Objects},
};

const MODEL: &str = "model-a";

/// The gateway both sides answer: the one that reaches the fleet's
/// host-local row.
const GATEWAY: GatewayId = GatewayId(7);

/// The fields that differ and are meant to agree, each with why it differs
/// today.
const DIFFERS: &[(&str, &str)] = &[
    (
        "candidate.engine_kind",
        "an object keeps a runtime the worker crate knows: a row's other name reads as none",
    ),
    (
        "candidate.host_local",
        "from where a row is reachable is not carried to its object",
    ),
    (
        "candidate.vision",
        "whether a row takes images is not carried to its object, and the writer is handed no source of it",
    ),
    (
        "candidate.wire_modes",
        "an object speaks one wire of the set its row states",
    ),
    (
        "demand.host_pipeline",
        "the writer keeps no demand column yet",
    ),
    ("demand.references", "the writer keeps no demand column yet"),
    ("demand.runtime", "the writer keeps no demand column yet"),
    ("demand.window", "the writer keeps no demand column yet"),
    (
        "outside.engines",
        "the store counts the workers whose engine family the document does not list: the writer drops the bit",
    ),
    (
        "outside.zones",
        "the store counts the workers whose zone the document does not list: the writer drops the bit",
    ),
    (
        "pool.reach",
        "the store hands each gateway its own list of a pool that holds a host-local row: the writer hands one list",
    ),
];

/// The fields that differ because each side numbers what it holds in its
/// own way. No step makes them agree: a caller that holds both sides pairs
/// their rows by [`Candidate::key`], as this case does.
const BY_DESIGN: &[(&str, &str)] = &[
    (
        "candidate.id",
        "the store numbers a worker's lives into its id: the writer names a worker by its address alone",
    ),
    (
        "candidate.position",
        "the store's rows stand in the order of their ids: the writer's in the order of the registry's list",
    ),
    (
        "pool.membership_epoch",
        "the store counts its writes: the writer counts the lists it made records of",
    ),
];

/// The fields the fleet leaves at rest on both sides, each with why. A line
/// names a field, or the start of several fields' names. The change that
/// sets one of them strikes its line, and the case then holds the field to
/// agree or to stand on a list above.
const AT_REST: &[(&str, &str)] = &[
    (
        "candidate.active_sessions",
        "no stage of the document counts a worker's sessions",
    ),
    ("candidate.connector", "the fleet holds no encoder"),
    (
        "candidate.dp_rank",
        "the fleet holds no engine of several ranks",
    ),
    ("candidate.ec_compatible", "the fleet holds no encoder"),
    (
        "candidate.health.breaker_open",
        "no worker's circuit breaker is open",
    ),
    (
        "candidate.health.overloaded",
        "no report is fed to either side yet",
    ),
    ("candidate.health.stalled", "no gateway vetoes a worker"),
    ("candidate.in_flight", "no request is on its way"),
    ("candidate.load", "no report is fed to either side yet"),
    (
        "candidate.pair_class",
        "the fleet holds no prefill and no decode",
    ),
    ("candidate.role", "the fleet is one pool of single workers"),
    ("candidate.session", "the decision carries no session key"),
    ("pool.engines", "the fleet holds no engine of several ranks"),
    ("pool.regime", "no report is fed to either side yet"),
    ("pool.reported", "no report is fed to either side yet"),
    ("pool.ring_epoch", "neither side keeps a ring of this pool"),
    ("pool.zones", "neither side counts a zone's capacity"),
];

/// The facts of a row that no row of the fleet states, each with why.
const NOT_STATED: &[(&str, &str)] = &[
    ("dp_rank", "the fleet holds no engine of several ranks"),
    (
        "role",
        "the fleet is one pool of single workers: a pair's and an encoder's are other pools",
    ),
];

/// A list of fields as they read: each one's name and its text.
type Fields = Vec<(&'static str, String)>;

/// Every field of a record, by name. The record, its health and its report
/// are taken apart with no rest pattern.
fn candidate_fields(row: &Candidate) -> Fields {
    let Candidate {
        position,
        id,
        role,
        dp_rank,
        dp_size,
        health,
        wire_modes,
        engine_kind,
        connector,
        in_flight,
        load,
        load_sampled_at_micros,
        session_hit,
        session_in_flight,
        pair_class,
        has_capacity,
        has_room,
        zone,
        labels,
        ec_compatible,
        vision,
        host_local,
        active_sessions,
        key,
    } = *row;
    let Health {
        healthy,
        breaker_open,
        stalled,
        draining,
        overloaded,
    } = health;
    let mut fields = vec![
        ("candidate.position", format!("{position:?}")),
        ("candidate.id", format!("{id:?}")),
        ("candidate.role", format!("{role:?}")),
        ("candidate.dp_rank", format!("{dp_rank:?}")),
        ("candidate.dp_size", format!("{dp_size:?}")),
        ("candidate.health.healthy", format!("{healthy:?}")),
        ("candidate.health.breaker_open", format!("{breaker_open:?}")),
        ("candidate.health.stalled", format!("{stalled:?}")),
        ("candidate.health.draining", format!("{draining:?}")),
        ("candidate.health.overloaded", format!("{overloaded:?}")),
        ("candidate.wire_modes", format!("{wire_modes:?}")),
        ("candidate.engine_kind", format!("{engine_kind:?}")),
        ("candidate.connector", format!("{connector:?}")),
        ("candidate.in_flight", format!("{in_flight:?}")),
        (
            "candidate.load_sampled_at_micros",
            format!("{load_sampled_at_micros:?}"),
        ),
        ("candidate.session_hit", format!("{session_hit:?}")),
        (
            "candidate.session_in_flight",
            format!("{session_in_flight:?}"),
        ),
        ("candidate.pair_class", format!("{pair_class:?}")),
        ("candidate.has_capacity", format!("{has_capacity:?}")),
        ("candidate.has_room", format!("{has_room:?}")),
        ("candidate.zone", format!("{zone:?}")),
        ("candidate.labels", format!("{labels:#b}")),
        ("candidate.ec_compatible", format!("{ec_compatible:?}")),
        ("candidate.vision", format!("{vision:?}")),
        ("candidate.host_local", format!("{host_local:?}")),
        ("candidate.active_sessions", format!("{active_sessions:?}")),
        ("candidate.key", format!("{key:#010x}")),
    ];
    fields.extend(load_fields(load));
    fields
}

/// Whether a record holds a report, and every field of the report by name.
/// A record that holds none reads each field as a report at rest does.
fn load_fields(load: Option<LoadRow>) -> Fields {
    let LoadRow {
        running,
        queued,
        waiting_uncached_tokens,
        kv_usage_permille,
        active_kv_permille,
        live_rate,
        resident_tokens,
        resident_counted,
        report_age_ms,
        source,
        max_running,
    } = load.unwrap_or_default();
    vec![
        ("candidate.load", format!("{:?}", load.is_some())),
        ("candidate.load.running", format!("{running:?}")),
        ("candidate.load.queued", format!("{queued:?}")),
        (
            "candidate.load.waiting_uncached_tokens",
            format!("{waiting_uncached_tokens:?}"),
        ),
        (
            "candidate.load.kv_usage_permille",
            format!("{kv_usage_permille:?}"),
        ),
        (
            "candidate.load.active_kv_permille",
            format!("{active_kv_permille:?}"),
        ),
        ("candidate.load.live_rate", format!("{live_rate:?}")),
        (
            "candidate.load.resident_tokens",
            format!("{resident_tokens:?}"),
        ),
        (
            "candidate.load.resident_counted",
            format!("{resident_counted:?}"),
        ),
        ("candidate.load.report_age_ms", format!("{report_age_ms:?}")),
        ("candidate.load.source", format!("{source:?}")),
        ("candidate.load.max_running", format!("{max_running:?}")),
    ]
}

/// Every fact a row states for a request's hard demands, by name.
fn demand_fields(stated: DemandFacts) -> Fields {
    let DemandFacts {
        runtime,
        references,
        host_pipeline,
        window,
    } = stated;
    vec![
        ("demand.runtime", format!("{runtime:?}")),
        ("demand.references", format!("{references:?}")),
        ("demand.host_pipeline", format!("{host_pipeline:?}")),
        ("demand.window", format!("{window:?}")),
    ]
}

/// Every fact of a pool beside its rows, by name.
fn pool_fields(pool: &PoolFacts<'_>) -> Fields {
    let PoolFacts {
        pool_size,
        membership_epoch,
        ring_epoch,
        regime,
        generation,
        zones,
        reach,
        reported,
        engines,
    } = *pool;
    vec![
        ("pool.pool_size", format!("{pool_size:?}")),
        ("pool.membership_epoch", format!("{membership_epoch:?}")),
        ("pool.ring_epoch", format!("{ring_epoch:?}")),
        ("pool.regime", format!("{regime:?}")),
        ("pool.generation", format!("{generation:?}")),
        ("pool.zones", format!("{zones:?}")),
        ("pool.reach", format!("{reach:?}")),
        ("pool.reported", format!("{reported:?}")),
        ("pool.engines", format!("{engines:?}")),
    ]
}

/// The store's count of the workers outside the document's lists, by name.
fn outside_fields(outside: OutsideLists) -> Fields {
    let OutsideLists { engines, zones } = outside;
    vec![
        ("outside.engines", format!("{engines:?}")),
        ("outside.zones", format!("{zones:?}")),
    ]
}

/// Every fact of a row by name, and whether `row` states it: whether it is
/// another value than a row that states nothing holds.
fn stated(row: &WorkerFacts) -> Vec<(&'static str, bool)> {
    let bare = WorkerFacts::default();
    let WorkerFacts {
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
        demand_facts,
    } = row;
    vec![
        ("key", *key != bare.key),
        ("engine", *engine != bare.engine),
        ("incarnation", *incarnation != bare.incarnation),
        ("role", *role != bare.role),
        ("wire_modes", *wire_modes != bare.wire_modes),
        ("engine_kind", *engine_kind != bare.engine_kind),
        ("models", *models != bare.models),
        ("dp_rank", *dp_rank != bare.dp_rank),
        ("dp_size", *dp_size != bare.dp_size),
        ("labels", *labels != bare.labels),
        ("zone", *zone != bare.zone),
        ("pairing", *pairing != bare.pairing),
        ("encode", *encode != bare.encode),
        ("vision", *vision != bare.vision),
        ("max_running", *max_running != bare.max_running),
        ("kv_block_size", *kv_block_size != bare.kv_block_size),
        ("bootstrap_port", *bootstrap_port != bare.bootstrap_port),
        ("priority", *priority != bare.priority),
        ("reachable_from", *reachable_from != bare.reachable_from),
        ("signals", *signals != bare.signals),
        ("health", *health != bare.health),
        ("overload", *overload != bare.overload),
        (
            "drain_settle_secs",
            *drain_settle_secs != bare.drain_settle_secs,
        ),
        ("demand_facts", *demand_facts != bare.demand_facts),
    ]
}

/// The document both sides read their rows under. It lists two zones, two
/// label selectors and two engine families, one of them a name the worker
/// crate does not know.
fn document() -> AppliedSpec {
    let mut document = PolicySpec::preset("least_load").expect("a built-in preset");
    document.generation = 9;
    document.view.zones = vec!["east".to_owned(), "west".to_owned()];
    document.view.labels = vec!["tier=gold".to_owned(), "rack".to_owned()];
    document.view.engines = vec!["generic".to_owned(), "house-engine".to_owned()];
    AppliedSpec::validate(&document).expect("the document validates")
}

/// The fleet: the rows of one model's pool of single workers, each with the
/// status it stands at. A row's width is one, as every registration writes
/// a single-rank worker's.
fn fleet() -> Vec<(WorkerFacts, WorkerStatus)> {
    let row = |key: &str| WorkerFacts {
        key: key.to_owned(),
        engine: key.to_owned(),
        role: Role::Single,
        wire_modes: WireModes::HTTP,
        models: vec![MODEL.to_owned()],
        dp_size: 1,
        ..WorkerFacts::default()
    };
    let labels = |pairs: &[(&str, &str)]| -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    };
    vec![
        // A worker whose registration states where it is and no more.
        (row("http://plain:8000"), WorkerStatus::Ready),
        // A worker that states every fact a row of one rank can, in the
        // zone the document lists first.
        (
            WorkerFacts {
                incarnation: 3,
                wire_modes: WireModes::GRPC,
                engine_kind: "generic".to_owned(),
                labels: labels(&[("tier", "gold"), ("rack", "a")]),
                zone: Some("east".to_owned()),
                pairing: Some(Arc::new(PairingDescriptor {
                    protocol: Some("p1".to_owned()),
                    transport: Some("t1".to_owned()),
                    version: Some("v1".to_owned()),
                    kv_layout: [Some("float16".to_owned()), None, None, None],
                })),
                encode: EncodeOffer {
                    connector: Some("c1".to_owned()),
                    ..EncodeOffer::default()
                },
                vision: true,
                max_running: Some(96),
                kv_block_size: Some(32),
                bootstrap_port: Some(8_998),
                priority: 4,
                signals: Signals {
                    events: true,
                    loads: true,
                    scrape_interval_ms: 3_000,
                },
                health: HealthPolicy {
                    failure_threshold: 4,
                    success_threshold: 5,
                    check_interval_ms: 7_000,
                    timeout_ms: 2_000,
                    probes: false,
                },
                overload: Overload {
                    kv_usage_permille: Some(800),
                    waiting_requests: Some(12),
                },
                drain_settle_secs: Some(30),
                demand_facts: DemandFacts {
                    runtime: 7,
                    references: true,
                    host_pipeline: true,
                    window: 8_192,
                },
                ..row("grpc://east:9000")
            },
            WorkerStatus::Ready,
        ),
        // Two wires, an engine family the worker crate does not know, and
        // the document's other zone. Not proven healthy yet.
        (
            WorkerFacts {
                incarnation: 5,
                wire_modes: WireModes::HTTP.union(WireModes::GRPC),
                engine_kind: "house-engine".to_owned(),
                labels: labels(&[("tier", "silver")]),
                zone: Some("west".to_owned()),
                demand_facts: DemandFacts {
                    runtime: 8,
                    references: true,
                    host_pipeline: false,
                    window: 4_096,
                },
                ..row("http://west:8000")
            },
            WorkerStatus::Pending,
        ),
        // A worker one gateway alone reaches, of an engine family and in a
        // zone the document does not list, on its way out.
        (
            WorkerFacts {
                incarnation: 2,
                engine_kind: "stray-engine".to_owned(),
                zone: Some("north".to_owned()),
                reachable_from: Reachability::Gateway(GATEWAY),
                ..row("http://local:8000")
            },
            WorkerStatus::Draining,
        ),
    ]
}

/// What one side hands a decision of the fleet's pool: each row's fields
/// under the row's key ([`Candidate::key`]), and the pool's. Beside them,
/// its count of the workers outside the document's lists.
struct Handed {
    rows: BTreeMap<u32, Fields>,
    pool: Fields,
    outside: Fields,
}

impl Handed {
    /// Every field the side reads, row after row, then the pool's and the
    /// count's.
    fn fields(&self) -> impl Iterator<Item = &(&'static str, String)> {
        let rows = self.rows.values().flatten();
        rows.chain(&self.pool).chain(&self.outside)
    }
}

/// The store's side: the view a decision of the pool is handed, with the
/// demand column beside its rows. The rows are asked for written, so each
/// carries its own bits and counts as a record of the writer does.
fn by_the_store(store: &MemoryStore, now: Instant) -> Handed {
    let world = store.snapshot();
    let ask = ViewQuery {
        gateway: GATEWAY,
        now,
        pinned: None,
        affinity: false,
        signal_cells: false,
    };
    let mut buffers = ViewBuffers::default();
    let view = world
        .view(MODEL, Role::Single, &ask, true, &mut buffers)
        .expect("the pool");
    let mut demanded = Vec::new();
    world.demand_facts(&view.built, &ask, &mut demanded);
    assert_eq!(demanded.len(), view.candidates.len(), "one entry a row");
    let rows = view.candidates.iter().zip(demanded).map(|(row, stated)| {
        let mut fields = candidate_fields(row);
        fields.extend(demand_fields(stated));
        (row.key, fields)
    });
    Handed {
        rows: rows.collect(),
        pool: pool_fields(&view.pool_facts()),
        outside: outside_fields(store.outside_lists()),
    }
}

/// The writer's side: the records of the registry's list, written for one
/// decision. The writer keeps no demand column and no count of the workers
/// outside the lists: every record states nothing for a demand, and the
/// count reads zero.
fn by_the_writer(registry: &WorkerRegistry<Members>, spec: &AppliedSpec, now: Instant) -> Handed {
    let workers = registry.get_by_model(MODEL);
    let clock = DecisionClock::new();
    let records = PoolRecords::new(&workers, spec, &clock);
    let live = Live::at(spec.engine(), &clock, now, None, None);
    let mut written = Vec::new();
    let (written, pool) = records.write(&workers, &live, &mut written);
    let rows = written.iter().map(|row| {
        let mut fields = candidate_fields(row);
        fields.extend(demand_fields(DemandFacts::default()));
        (row.key, fields)
    });
    Handed {
        rows: rows.collect(),
        pool: pool_fields(&pool),
        outside: outside_fields(OutsideLists::default()),
    }
}

/// The fields that read otherwise on the two sides: each one's name, with
/// the first place it does and both texts.
fn differing(stored: &Handed, written: &Handed) -> BTreeMap<&'static str, String> {
    let mut found = BTreeMap::new();
    let mut compare = |at: &str, stored: &Fields, written: &Fields| {
        assert_eq!(stored.len(), written.len(), "{at}: one list of names");
        for ((name, stored), (same, written)) in stored.iter().zip(written) {
            assert_eq!(name, same, "{at}: one order of names");
            if stored != written {
                found
                    .entry(*name)
                    .or_insert_with(|| format!("{at}: the store {stored}, the writer {written}"));
            }
        }
    };
    for (key, fields) in &stored.rows {
        compare(&format!("the row {key:#010x}"), fields, &written.rows[key]);
    }
    compare("the pool", &stored.pool, &written.pool);
    compare("the count", &stored.outside, &written.outside);
    found
}

/// The fields that read, on every row of both sides, as they do on a record
/// no host has written: what the fleet leaves at rest.
fn resting(sides: [&Handed; 2]) -> BTreeSet<&'static str> {
    let mut rest = candidate_fields(&Candidate::default());
    rest.extend(demand_fields(DemandFacts::default()));
    rest.extend(pool_fields(&PoolFacts::default()));
    rest.extend(outside_fields(OutsideLists::default()));
    let mut resting: BTreeSet<&'static str> = rest.iter().map(|(name, _)| *name).collect();
    for side in sides {
        for read in side.fields() {
            if !rest.contains(read) {
                resting.remove(read.0);
            }
        }
    }
    resting
}

/// The names a list holds, each once.
fn names(list: &'static [(&'static str, &'static str)]) -> BTreeSet<&'static str> {
    let names: BTreeSet<&'static str> = list.iter().map(|(name, _)| *name).collect();
    assert_eq!(names.len(), list.len(), "a list names a field once");
    names
}

/// One fleet, handed to a decision by the store and by the record writer
/// over the objects the worker host lays of it: every field that differs
/// stands on a list with why, and so does every field the fleet leaves at
/// rest.
#[test]
fn the_writers_records_differ_from_the_stores_view_in_the_listed_fields_alone() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    store.apply(document()).expect("the first generation");
    for (facts, status) in fleet() {
        let id = store.upsert(facts).id();
        let revision = store.get(id).expect("the row").revision;
        // A worker on its way out was ready first.
        let through = (status == WorkerStatus::Draining).then_some(WorkerStatus::Ready);
        for status in through.into_iter().chain([status]) {
            store.set_status(id, status, revision);
        }
        assert_eq!(store.get(id).expect("the row").status, status);
    }
    store.publish_now();

    let registry = Arc::new(WorkerRegistry::<Members>::new());
    let objects = Objects::new(Arc::clone(&registry));
    objects.follow(&store);
    assert_eq!(objects.len(), fleet().len(), "one object a row");

    let now = Instant::now();
    let stored = by_the_store(&store, now);
    let written = by_the_writer(&registry, &document(), now);
    let keys = |side: &Handed| side.rows.keys().copied().collect::<Vec<_>>();
    assert_eq!(
        keys(&stored).len(),
        fleet().len(),
        "the gateway is handed every row"
    );
    assert_eq!(
        keys(&stored),
        keys(&written),
        "both sides hold the same rows"
    );

    let mut wrong = Vec::new();

    // The rows: every fact is stated by one of them, but the listed ones.
    let mut never: BTreeSet<&'static str> = BTreeSet::new();
    for (facts, _) in fleet() {
        never.extend(stated(&facts).into_iter().map(|(name, _)| name));
    }
    for (facts, _) in fleet() {
        for (name, _) in stated(&facts).into_iter().filter(|(_, stated)| *stated) {
            never.remove(name);
        }
    }
    if never != names(NOT_STATED) {
        wrong.push(format!(
            "the facts no row states are {never:?}, and NOT_STATED lists {:?}",
            names(NOT_STATED)
        ));
    }

    // The fields that differ: each on one list, and each list's fields
    // differ.
    let found = differing(&stored, &written);
    let (differs, by_design) = (names(DIFFERS), names(BY_DESIGN));
    assert!(
        differs.is_disjoint(&by_design),
        "a field stands on one list"
    );
    for (name, at) in &found {
        if !differs.contains(name) && !by_design.contains(name) {
            wrong.push(format!("{name} differs and stands on no list ({at})"));
        }
    }
    for name in differs.iter().chain(&by_design) {
        if !found.contains_key(name) {
            wrong.push(format!(
                "{name} is listed as differing and agrees: strike its line"
            ));
        }
    }

    // The fields at rest: each under one line, and each line's fields rest.
    let rest = resting([&stored, &written]);
    let lines = names(AT_REST);
    let line_of = |name: &str| lines.iter().find(|line| name.starts_with(**line)).copied();
    for name in &rest {
        if line_of(name).is_none() {
            wrong.push(format!(
                "{name} rests on both sides and AT_REST has no line for it"
            ));
        }
    }
    let every: BTreeSet<&'static str> = stored.fields().map(|(name, _)| *name).collect();
    for name in every.difference(&rest) {
        if let Some(line) = line_of(name) {
            wrong.push(format!(
                "{name} is set and AT_REST lists it under {line}: strike or narrow the line"
            ));
        }
    }
    for line in &lines {
        if !every.iter().any(|name| name.starts_with(*line)) {
            wrong.push(format!(
                "AT_REST lists {line} and no field's name starts so"
            ));
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}
