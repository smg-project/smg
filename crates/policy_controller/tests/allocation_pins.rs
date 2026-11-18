//! What a decision allocates, pinned at zero: once a first decision has sized
//! the per-thread scratch, a decision of every preset over a reporting fleet
//! of 64 and of 512 rows, a decision over rows with a credit held in the
//! least-load tally, and a pair decision allocate nothing (design 4.4: a
//! scorer or picker allocates nothing per decision; the host keeps one scratch
//! per thread so the framework allocates nothing after the first). The
//! store's pins stand beside them: a build from the store allocates nothing
//! after the first, and two membership writes are held to each other, a
//! write that moves an offer to one that moves none and a registration into
//! 512 workers to one into 8. A separate executable, so it can declare the
//! counting allocator the pins need.

#![expect(clippy::expect_used, reason = "tests that stop on a broken setup")]

mod common;

use std::time::{Duration, Instant};

use common::counting_alloc::{calls, Counting};
use smg_policy_controller::{
    store::{
        Ask, GatewayId, HealthSink, LoadSink, LoadSource, MemoryStore, PairBuffers,
        PairingDescriptor, ViewBuffers, WorkerFacts, WorkerStatus, WorkersStore, PUBLISH_WINDOW,
    },
    AffinityFacts, Candidate, Constraints, Engine, Health, Lease, Lifecycle, LoadRow,
    NamedAffinity, NamedHolder, PairPools, PolicySpec, PoolFacts, RequestFacts, Role, Scratch,
    Side, SignalSource, Target, WireModes, WorkerId, PRESETS,
};

#[global_allocator]
static GLOBAL: Counting = Counting;

/// What `work` returns, and the allocation calls this thread made during it.
fn allocations_during<T>(work: impl FnOnce() -> T) -> (T, usize) {
    let before = calls();
    let out = work();
    (out, calls() - before)
}

/// A reporting fleet: every sixth row unhealthy, loads spread, as a host
/// reads its pool between two reports.
fn fleet(n: u32) -> Vec<Candidate> {
    (0..n)
        .map(|i| Candidate {
            position: i,
            id: WorkerId(1_000 + u64::from(i)),
            health: Health {
                healthy: i % 6 != 5,
                ..Health::default()
            },
            in_flight: i % 7,
            load: Some(LoadRow {
                running: i % 5,
                queued: i % 3,
                kv_usage_permille: (i * 37 % 900) as u16,
                live_rate: 1_000 + i * 13 % 3_000,
                report_age_ms: 5,
                source: SignalSource::Events,
                ..LoadRow::default()
            }),
            load_sampled_at_micros: 990_000 + u64::from(i % 9) * 1_000,
            ..Candidate::default()
        })
        .collect()
}

fn engine_of(preset: &str) -> Engine {
    PolicySpec::preset(preset)
        .expect("a preset of this build")
        .validate()
        .expect("a preset validates")
}

fn facts(seed: u64) -> RequestFacts<'static> {
    RequestFacts {
        model: "m",
        prompt_tokens: 512,
        seed,
        now_micros: 1_000_000,
        ..RequestFacts::default()
    }
}

fn pool(n: u32) -> PoolFacts<'static> {
    PoolFacts {
        pool_size: n,
        generation: 1,
        ..PoolFacts::default()
    }
}

#[test]
fn a_decision_after_the_first_allocates_nothing_for_any_preset() {
    for preset in PRESETS {
        let engine = engine_of(preset);
        for n in [64u32, 512] {
            let view = fleet(n);
            let cx = Constraints::default();
            let mut scratch = Scratch::new();
            engine
                .decide(&facts(1), &pool(n), &view, &cx, &mut scratch)
                .pick()
                .expect("the fleet routes");
            let (decision, allocations) =
                allocations_during(|| engine.decide(&facts(2), &pool(n), &view, &cx, &mut scratch));
            assert!(decision.pick().is_some(), "{preset} at {n} routes");
            assert_eq!(allocations, 0, "{preset} at {n}: a decision allocated");
        }
    }
}

/// A decision asked by rank allocates nothing after the thread's first of
/// its size, whatever that first was: by rank where no row holds the
/// request's prefix, a draw for a request that names rows, or the whole
/// pass over a pool not said to hold a rate on every row. The second
/// decision puts the holders in front: one row in eight, each running one
/// request at a rate a token a second from the next, with a long queue on
/// one row behind them. Its targets are the four fastest holders in their
/// order, which shows the front: the whole pass spreads the load over that
/// queue, reads the holders alike and draws among them. What the decision
/// keeps of the rows in front stands in buffers every first decision
/// sized: their sums in the stage's, their list in the soft filter's
/// backup. The preset is read with its soft filters and without them. A
/// profile without one takes the list's room at its first decision by
/// rank: behind a whole pass that decision allocates once, and the next
/// allocates nothing.
#[test]
fn a_decision_by_rank_after_the_first_allocates_nothing() {
    let mut bare = PolicySpec::preset("cache_aware").expect("a preset");
    for profile in bare.profiles.values_mut() {
        profile
            .filter
            .retain(|name| name != "not-tried" && name != "has-room");
    }
    let engines = [
        (true, engine_of("cache_aware")),
        (false, bare.validate().expect("the document validates")),
    ];
    for (soft, engine, n) in engines
        .iter()
        .flat_map(|(soft, engine)| [(*soft, engine, 64u32), (*soft, engine, 512)])
    {
        let mut view = fleet(n);
        for (step, row) in (0u32..).zip(view.iter_mut().step_by(8)) {
            row.in_flight = 0;
            if let Some(load) = row.load.as_mut() {
                load.running = 1;
                load.queued = 0;
                load.kv_usage_permille = 100;
                load.live_rate = 20_000 + step;
            }
        }
        if let Some(load) = view[1].load.as_mut() {
            load.queued = 200;
        }
        let fastest = [n - 8, n - 16, n - 24, n - 32];
        let column = |blocks: u32| -> Vec<AffinityFacts> {
            (0..n)
                .map(|i| AffinityFacts {
                    overlap_blocks: if i % 8 == 0 { blocks } else { 0 },
                    ..AffinityFacts::default()
                })
                .collect()
        };
        let (cold, held) = (column(0), column(4));
        let names = [NamedHolder {
            row: 0,
            overlap_blocks: 4,
            predicted_blocks: 0,
        }];
        // Each first, and whether the host says the pool's fact of it.
        let firsts = [
            (
                RequestFacts {
                    affinity: &cold,
                    ..facts(1)
                },
                true,
            ),
            (
                RequestFacts {
                    named: Some(NamedAffinity {
                        holders: &names,
                        ..NamedAffinity::default()
                    }),
                    ..facts(1)
                },
                true,
            ),
            (
                RequestFacts {
                    affinity: &cold,
                    ..facts(1)
                },
                false,
            ),
        ];
        let said = PoolFacts {
            rated: true,
            ..pool(n)
        };
        let unsaid = pool(n);
        let cx = Constraints::default();
        for (order, (first, rated)) in firsts.iter().enumerate() {
            let mut scratch = Scratch::new();
            let of_first = if *rated { &said } else { &unsaid };
            engine
                .decide(first, of_first, &view, &cx, &mut scratch)
                .pick()
                .expect("the fleet routes");
            let second = RequestFacts {
                affinity: &held,
                ..facts(2)
            };
            let (decision, allocations) =
                allocations_during(|| engine.decide(&second, &said, &view, &cx, &mut scratch));
            let pick = decision.pick().expect("the fleet routes");
            let targets: Vec<u32> = pick.targets.iter().map(|target| target.position).collect();
            assert_eq!(
                targets, fastest,
                "soft {soft} at {n}, order {order}: the fastest holders, in their order"
            );
            // A whole pass sizes no list of the rows in front: without a
            // soft filter the first decision by rank takes that room.
            let lists_late = !soft && !*rated;
            assert_eq!(
                allocations,
                usize::from(lists_late),
                "soft {soft} at {n}, order {order}: a decision by rank allocated"
            );
            let (_, again) =
                allocations_during(|| engine.decide(&second, &said, &view, &cx, &mut scratch));
            assert_eq!(
                again, 0,
                "soft {soft} at {n}, order {order}: the next decision by rank allocated"
            );
        }
    }
}

#[test]
fn a_decision_after_a_soft_filter_fallback_allocates_nothing() {
    // Every healthy row tried: the soft not-tried stage empties the
    // survivors and the stage before it is restored. The decision after
    // that fallback finds its scratch sized still.
    let engine = engine_of("round_robin");
    let view = fleet(64);
    let tried: Vec<WorkerId> = view
        .iter()
        .filter(|c| c.health.healthy)
        .map(|c| c.id)
        .collect();
    let first = RequestFacts {
        tried: &tried,
        ..facts(1)
    };
    let next = RequestFacts {
        tried: &tried,
        ..facts(2)
    };
    let cx = Constraints::default();
    let mut scratch = Scratch::new();
    engine
        .decide(&first, &pool(64), &view, &cx, &mut scratch)
        .pick()
        .expect("the restored survivors route");
    let (decision, allocations) =
        allocations_during(|| engine.decide(&next, &pool(64), &view, &cx, &mut scratch));
    assert!(decision.pick().is_some());
    assert_eq!(allocations, 0, "the decision after a fallback allocated");
}

#[test]
fn a_round_robin_decision_over_a_changing_survivor_set_allocates_nothing() {
    // One tried exclusion rotating through the pool makes every survivor set
    // new to the rotation table: its claim path must not allocate.
    let engine = engine_of("round_robin");
    for n in [64u32, 512] {
        let view = fleet(n);
        let cx = Constraints::default();
        let mut scratch = Scratch::new();
        let mut churn = |seed: u64| {
            let tried = [view[(seed % u64::from(n)) as usize].id];
            let f = RequestFacts {
                tried: &tried,
                ..facts(seed)
            };
            engine
                .decide(&f, &pool(n), &view, &cx, &mut scratch)
                .pick()
                .is_some()
        };
        // The first decisions size the scratch for the largest survivor set
        // the pin will see (an unhealthy row excluded leaves one more
        // survivor than a healthy one excluded).
        assert!(churn(5) && churn(1), "round_robin churn at {n} routes");
        let (routed, allocations) = allocations_during(|| (2..66u64).all(&mut churn));
        assert!(routed, "round_robin churn at {n} routes");
        assert_eq!(
            allocations, 0,
            "round_robin churn at {n}: a decision allocated"
        );
    }
}

#[test]
fn the_pass_behind_a_draw_that_keeps_nobody_allocates_nothing() {
    // The first decision draws over a pool with every row ready and writes
    // a few rows. A later one meets the pool with one row ready: the draw
    // keeps nobody and the whole pass runs, for the first time on this
    // scratch. The first decision took the pass's room.
    for preset in ["passthrough", "random", "round_robin", "power_of_two"] {
        let engine = engine_of(preset);
        let mut ready = fleet(512);
        for row in &mut ready {
            row.health.healthy = true;
        }
        let mut lone = ready.clone();
        for row in &mut lone {
            row.health.healthy = row.position == 300;
        }
        let cx = Constraints::default();
        let mut scratch = Scratch::new();
        let first = engine.decide(&facts(1), &pool(512), &ready, &cx, &mut scratch);
        assert!(
            first.trace().drew_first(),
            "{preset}: the first decision drew"
        );
        let mut widened = 0;
        for seed in 2..66u64 {
            let (decision, allocations) = allocations_during(|| {
                engine.decide(&facts(seed), &pool(512), &lone, &cx, &mut scratch)
            });
            assert_eq!(allocations, 0, "{preset}: a decision allocated");
            let pick = decision.pick().expect("the lone row routes");
            assert_eq!(pick.targets.first().map(|t| t.position), Some(300));
            widened += u32::from(!pick.trace.drew_first());
        }
        assert!(widened > 0, "{preset}: no decision reached the whole pass");
    }
}

#[test]
fn a_decision_over_credited_rows_allocates_nothing() {
    let engine = engine_of("least_load");
    let view = fleet(512);
    let lease = Lease {
        issued_micros: 1_000_000,
        tokens: 512,
        ..Lease::default()
    };
    for c in &view {
        let target = Target {
            position: c.position,
            id: c.id,
            dp_rank: c.dp_rank,
            pair: None,
        };
        engine.on_dispatched(&lease, &target, Side::Single, 0);
    }
    let cx = Constraints::default();
    let mut scratch = Scratch::new();
    engine.decide(&facts(1), &pool(512), &view, &cx, &mut scratch);
    let (decision, allocations) =
        allocations_during(|| engine.decide(&facts(2), &pool(512), &view, &cx, &mut scratch));
    assert!(decision.pick().is_some());
    assert_eq!(
        allocations, 0,
        "a decision reading the credit tally allocated"
    );
}

#[test]
fn a_pair_decision_under_a_table_allocates_nothing_after_the_first() {
    let engine = PolicySpec::pair_presets("least_load", "least_load")
        .expect("the pair spec")
        .validate()
        .expect("the engine");
    let mut prefills = fleet(16);
    for p in &mut prefills {
        p.pair_class = p.position % 4 + 1;
        p.has_capacity = true;
    }
    let mut decodes = fleet(64);
    for d in &mut decodes {
        d.pair_class = d.position % 8 + 1;
        d.has_room = true;
    }
    // Prefill class k pairs with decode classes k and k + 4.
    let (ends, classes) = ([2, 4, 6, 8], [1, 5, 2, 6, 3, 7, 4, 8]);
    let cx = Constraints::default();
    let mut scratch = Scratch::new();
    let mut decide = |decodes: &[Candidate], seed: u64| {
        let pools = PairPools {
            prefills: &prefills,
            decodes,
            table: smg_policy_controller::PairTable::new(&ends, &classes),
            ..PairPools::default()
        };
        allocations_during(|| {
            engine.decide_pair(&facts(seed), &pool(80), pools, &cx, &cx, &mut scratch)
        })
    };
    // A pair, then a shed that reads the routable prefills' lists: each
    // sizes the scratch once.
    assert!(decide(&decodes, 1).0.pick().is_some(), "the pair routes");
    let (decision, allocations) = decide(&decodes, 2);
    assert!(decision.pick().is_some());
    assert_eq!(allocations, 0, "a pair decision under a table allocated");
    for d in &mut decodes {
        d.health.healthy = false;
    }
    assert!(decide(&decodes, 3).0.pick().is_none(), "no decode is up");
    let (decision, allocations) = decide(&decodes, 4);
    assert!(decision.pick().is_none());
    assert_eq!(allocations, 0, "a pair shed under a table allocated");
}

#[test]
fn a_pair_decision_after_the_first_allocates_nothing() {
    let engine = PolicySpec::pair_presets("least_load", "least_load")
        .expect("the pair spec")
        .validate()
        .expect("the engine");
    let mut prefills = fleet(16);
    for p in &mut prefills {
        p.pair_class = p.position % 4 + 1;
        p.has_capacity = true;
    }
    let mut decodes = fleet(64);
    for d in &mut decodes {
        d.pair_class = d.position % 4 + 1;
        d.has_room = true;
    }
    let cx = Constraints::default();
    let mut scratch = Scratch::new();
    let pools = PairPools {
        prefills: &prefills,
        decodes: &decodes,
        ..PairPools::default()
    };
    engine
        .decide_pair(&facts(1), &pool(80), pools, &cx, &cx, &mut scratch)
        .pick()
        .expect("the pair routes");
    let (decision, allocations) = allocations_during(|| {
        engine.decide_pair(&facts(2), &pool(80), pools, &cx, &cx, &mut scratch)
    });
    assert!(decision.pick().is_some());
    assert_eq!(allocations, 0, "a pair decision allocated");
}

/// A store of two pools: `n` ready members serving `m` and one member
/// serving another model, each with a record on file and published. The
/// members of `m` and the instant the store's last window opened.
fn store_of(n: usize) -> (MemoryStore, Vec<WorkerId>, Instant) {
    let store = MemoryStore::new();
    let opened = Instant::now();
    let register = |key: String, model: &str| {
        let id = store
            .upsert(WorkerFacts {
                key: key.clone(),
                engine: key,
                incarnation: 1,
                models: vec![model.to_string()],
                dp_size: 1,
                ..WorkerFacts::default()
            })
            .id();
        let revision = WorkersStore::get(&store, id).expect("the row").revision;
        store
            .set_status(id, WorkerStatus::Ready, revision)
            .expect("ready");
        store.record(id, 0, LoadRow::default(), opened, LoadSource::Stream);
        id
    };
    let ids = (0..n)
        .map(|index| register(format!("http://w{index}:8000"), "m"))
        .collect();
    register("http://other:8000".to_string(), "other");
    // The records that landed inside the first one's window are published
    // at its end: nothing waits.
    let opened = opened + PUBLISH_WINDOW;
    store.publish(opened);
    assert_eq!(store.publish_due(), None);
    (store, ids, opened)
}

#[test]
fn a_build_from_the_store_after_the_first_allocates_nothing() {
    for n in [8usize, 64, 512] {
        let (store, _, opened) = store_of(n);
        let mut out = ViewBuffers::default();
        for affinity in [false, true] {
            let ask = Ask {
                gateway: GatewayId(1),
                now: opened + Duration::from_secs(1),
                pinned: None,
                affinity,
            };
            // The first build sizes the host's buffers.
            store
                .build_view("m", Role::Single, &ask, &mut out)
                .expect("the pool");
            let (built, allocations) =
                allocations_during(|| store.build_view("m", Role::Single, &ask, &mut out));
            assert_eq!(built.map(|built| built.pool_size as usize), Some(n));
            assert_eq!(out.affinity.len(), if affinity { n } else { 0 });
            assert_eq!(
                allocations, 0,
                "{n} members, column {affinity}: a build allocated"
            );
        }
    }
}

#[test]
fn a_pair_build_with_its_table_allocates_nothing_after_the_first() {
    // Two prefills of two transports; of 64 decodes every fourth names no
    // transport and pairs with both: three decode classes in two lists.
    let store = MemoryStore::new();
    let register = |key: String, role: Role, transport: Option<&str>| {
        store.upsert(WorkerFacts {
            key: key.clone(),
            engine: key,
            incarnation: 1,
            role,
            wire_modes: WireModes::GRPC,
            models: vec!["m".to_string()],
            dp_size: 1,
            pairing: transport.map(|transport| {
                Box::new(PairingDescriptor {
                    transport: Some(transport.to_string()),
                    ..PairingDescriptor::default()
                })
            }),
            ..WorkerFacts::default()
        });
    };
    for (index, transport) in ["tx-a", "tx-b"].into_iter().enumerate() {
        register(
            format!("grpc://p{index}:9000"),
            Role::Prefill,
            Some(transport),
        );
    }
    for index in 0..64 {
        let transport = [Some("tx-a"), Some("tx-b"), Some("tx-a"), None][index % 4];
        register(format!("grpc://d{index}:9000"), Role::Decode, transport);
    }
    let ask = Ask {
        gateway: GatewayId(1),
        now: Instant::now(),
        pinned: None,
        affinity: true,
    };
    let mut out = PairBuffers::default();
    // The first build sizes the host's buffers: the rows, the columns and
    // the table.
    store.build_pair("m", &ask, &mut out).expect("both pools");
    let (built, allocations) = allocations_during(|| store.build_pair("m", &ask, &mut out));
    let sizes = built.map(|(prefill, decode)| (prefill.pool_size, decode.pool_size));
    assert_eq!(sizes, Some((2, 64)));
    let pools = out.pools();
    assert_eq!(
        (pools.prefill_affinity.len(), pools.decode_affinity.len()),
        (2, 64)
    );
    // Each prefill class lists its transport's decodes and the silent ones.
    let world = store.world();
    let classes = world.pairs("m").expect("the model's classes");
    assert_eq!((classes.classes(), classes.entries()), (2, 4));
    assert_eq!(pools.table.validate(), Ok(()));
    assert_eq!(allocations, 0, "a pair build through the door allocated");
}

#[test]
fn a_write_that_moves_an_offer_allocates_what_a_write_that_moves_none_does() {
    // Pairs that name no runtime. One prefill names its runtime and unnames
    // it in turn, which moves its offer at every write and no class: the
    // writer derives for that worker alone from what it keeps, and hands
    // back the classes it had.
    for n in [8_usize, 64] {
        let store = MemoryStore::new();
        let facts_of = |index: usize, runtime: &str| {
            let (role, key) = if index < n {
                (Role::Prefill, format!("grpc://p{index}:9000"))
            } else {
                (Role::Decode, format!("grpc://d{index}:9000"))
            };
            WorkerFacts {
                key: key.clone(),
                engine: key,
                incarnation: 1,
                role,
                wire_modes: WireModes::GRPC,
                engine_kind: runtime.to_string(),
                models: vec!["m".to_string()],
                dp_size: 1,
                ..WorkerFacts::default()
            }
        };
        for index in 0..2 * n {
            store.upsert(facts_of(index, ""));
        }
        // The fewest allocation calls of one write among a few: the store's
        // change log grows by doubling, and one write in many pays for it.
        let fewest = |runtimes: [&str; 2]| {
            let one = |turn: usize| {
                let facts = facts_of(n / 2, runtimes[turn % 2]);
                allocations_during(|| store.upsert(facts)).1
            };
            (0..8).map(one).min()
        };
        // The first writes size what the writer keeps of the derivation.
        fewest(["rt-b", ""]);
        let (kept, moved) = (fewest(["", ""]), fewest(["rt-b", ""]));
        assert_eq!(
            moved, kept,
            "{n} a side: the written worker's derive allocated"
        );
    }
}

#[test]
fn a_registrations_allocations_do_not_follow_the_fleet() {
    // A membership write walks every member into its pool's list. A
    // standing pool's list starts at the pool's last length and one more,
    // so what a registration allocates does not follow the fleet: one into
    // 512 workers makes the allocator calls of one into 8.
    let fewest = |n: usize| {
        let (store, _, _) = store_of(n);
        // The fewest allocation calls of one registration among a few: the
        // store's tables and the writer's kept lists grow by doubling, and
        // one write in many pays for it.
        let one = |turn: usize| {
            let key = format!("http://new{turn}:8000");
            let facts = WorkerFacts {
                key: key.clone(),
                engine: key,
                incarnation: 1,
                models: vec!["m".to_string()],
                dp_size: 1,
                ..WorkerFacts::default()
            };
            allocations_during(|| store.upsert(facts)).1
        };
        (0..8).map(one).min()
    };
    let (few, many) = (fewest(8), fewest(512));
    assert_eq!(many, few, "a registration's allocations follow the fleet");
}

#[test]
fn a_record_inside_the_window_allocates_nothing_and_a_publication_derives_nothing() {
    let (store, ids, opened) = store_of(64);
    // Inside the window the store's making opened: on file and folded,
    // nothing published.
    let inside = opened + Duration::from_millis(1);
    let record = LoadRow {
        running: 7,
        ..LoadRow::default()
    };
    let ((), allocations) =
        allocations_during(|| store.record(ids[0], 0, record, inside, LoadSource::Stream));
    assert_eq!(allocations, 0, "a load record inside the window allocated");
    // The window's end: the rows of the one pool a record was folded into,
    // the world and its pointer a pool. Nothing the membership derived is
    // built again, and the other pool's rows are not copied.
    let due = store.publish_due().expect("the record waits");
    let ((), allocations) = allocations_during(|| store.publish(due));
    assert_eq!(allocations, 3, "a publication of one pool's rows");
    assert_eq!(store.publish_due(), None);
}

/// A validation builds an index key's error path at the key's refusal
/// alone: keys that pass add no allocation, so a spec of a thousand keys
/// more allocates nothing more.
#[test]
fn index_keys_that_pass_add_no_allocation_to_a_validation() {
    // A process's first validation builds the stock catalog, once: unmeasured.
    let first = PolicySpec::preset("round_robin").expect("a preset");
    assert!(first.validate().is_ok(), "the preset validates");
    let allocations = |keys: u32| {
        let mut spec = PolicySpec::preset("round_robin").expect("a preset");
        for key in 0..keys {
            spec.index
                .insert(format!("model-{key:04}*"), Default::default());
        }
        let (engine, allocations) = allocations_during(|| spec.validate());
        assert!(engine.is_ok(), "{keys} keys validate");
        allocations
    };
    assert_eq!(allocations(2_000), allocations(1_000));
}

#[test]
fn a_lent_view_from_the_store_after_the_first_allocates_nothing() {
    for n in [8usize, 64, 512] {
        let (store, ids, opened) = store_of(n);
        let asker = GatewayId(1);
        store.gateway_veto(asker, ids[1], Some(opened + Duration::from_secs(3_600)));
        let mut out = ViewBuffers::default();
        for affinity in [false, true] {
            let ask = Ask {
                gateway: asker,
                now: opened + Duration::from_secs(1),
                pinned: Some(ids[0]),
                affinity,
            };
            // The first view sizes the host's buffers: the mask and the
            // column.
            store.with_world(|world| {
                let first = world.view("m", Role::Single, &ask, false, &mut out);
                assert!(first.is_some());
            });
            let (read, allocations) = allocations_during(|| {
                store.with_world(|world| {
                    let view = world
                        .view("m", Role::Single, &ask, false, &mut out)
                        .expect("the pool");
                    let (rows, column) = (view.candidates.len(), view.affinity.len());
                    (rows, view.stalled.len(), view.pinned_at != 0, column)
                })
            });
            assert_eq!(read, (n, 1, true, if affinity { n } else { 0 }));
            assert_eq!(
                allocations, 0,
                "a lent view of {n} rows allocated (affinity {affinity})"
            );
        }
        assert!(out.candidates.is_empty(), "a lent view copies no row");
    }
}
