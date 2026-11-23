//! What a decision allocates, pinned at zero: once a first decision has sized
//! the per-thread scratch, a decision of every preset over a reporting fleet
//! of 64 and of 512 rows, a least-cost decision over mixed reports and sends, a
//! decision over rows with a credit held in the least-load tally, a decision
//! over a pool doubted on every row, and a pair decision allocate nothing
//! (design 4.4: a scorer or picker allocates nothing per decision; the host
//! keeps one scratch per thread so the framework allocates nothing after the
//! first). The
//! store's pins stand beside them: two membership writes are held to each
//! other, a write that moves an offer to one that moves none and a
//! registration into 512 workers to one into 8. A lease's life at the store
//! allocates nothing once its file is sized: its record filed, its
//! dispatch and its completion handed to the file, the completion written
//! for the health rule and the sweep's pass. A host in process is pinned
//! too, in a build
//! with the `in-process` feature: a request held from its decision to its
//! completion allocates nothing after the first ones. A separate
//! executable, so it can declare the counting allocator the pins need.

#![expect(clippy::expect_used, reason = "tests that stop on a broken setup")]

mod common;

use std::time::{Duration, Instant};

use common::counting_alloc::{calls, Counting};
use smg_policy_controller::{
    document::{PolicySpec, PRESETS},
    engine::{
        AffinityFacts, AffinityHolder, AffinityRows, Candidate, Constraints, Engine, Health,
        LoadRow, PairPools, PoolFacts, RequestFacts, Role, Scratch, SignalSource, Target,
        WireModes, WorkerId,
    },
    stages::Lifecycle,
    store::{
        GatewayId, Hand, HealthSink, LeaseRecord, LeaseState, LeasesStore, LoadSink, LoadSource,
        MemoryStore, PickRecord, WorkerFacts, WorkerStatus, WorkersStore, PUBLISH_WINDOW,
    },
    tracking::{Dispatch, Lease, LeaseId, Outcome, Said, Side, TransportClass},
};

#[global_allocator]
static GLOBAL: Counting = Counting(std::alloc::System);

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

#[test]
fn a_least_cost_decision_after_the_first_allocates_nothing() {
    let profile = serde_yaml::from_str(
        "filter: [healthy]\nscore: [{least-cost: {terms: {load.running: 7, load.queued: 13, sent.requests: 3, sent.tokens: 1}, unreported: {over-least: {request_cost: 11}}}}]\npick: best",
    )
    .expect("a least-cost profile");
    let engine = PolicySpec::new([("p".to_string(), profile)].into())
        .validate()
        .expect("the profile validates");
    let n = 512;
    let mut view = fleet(n);
    for (i, row) in (0u32..).zip(&mut view) {
        row.health.healthy = true;
        if i.is_multiple_of(3) {
            row.load = None;
        }
    }
    // Every row holds a send later than its report.
    let lease = Lease {
        issued_micros: 1_000_000,
        tokens: 512,
        ..Lease::default()
    };
    for row in &view {
        let target = Target {
            position: row.position,
            id: row.id,
            dp_rank: row.dp_rank,
            pair: None,
        };
        engine.on_dispatched(&lease, &target, Side::Single, 0);
    }
    let cx = Constraints::default();
    let mut scratch = Scratch::new();
    let pool = pool(n);
    assert!(engine
        .decide(&facts(1), &pool, &view, &cx, &mut scratch)
        .pick()
        .is_some());
    let (decision, allocations) =
        allocations_during(|| engine.decide(&facts(2), &pool, &view, &cx, &mut scratch));
    assert!(decision.pick().is_some());
    assert_eq!(allocations, 0);
}

/// `has-room` by a rule reads every row's report and its sends. A decision
/// after the first allocates nothing at 512 rows, under a picker that
/// reads the whole pool and under one that draws first.
#[test]
fn a_decision_under_a_has_room_rule_allocates_nothing_after_the_first() {
    for picker in ["round-robin", "random"] {
        let profile = serde_yaml::from_str(&format!(
            "filter: [healthy, {{has-room: {{used: {{load.running: 1, load.queued: 1, request.prompt_tokens: 1, sent.tokens: 1}}, capacity: 2000, headroom: 0.05}}}}]\npick: {picker}",
        ))
        .expect("a profile with the rule");
        let engine = PolicySpec::new([("p".to_string(), profile)].into())
            .validate()
            .expect("the profile validates");
        let n = 512;
        let mut view = fleet(n);
        for (i, row) in (0u32..).zip(&mut view) {
            row.health.healthy = true;
            if i.is_multiple_of(3) {
                row.load = None;
            }
        }
        // Every row holds a send later than its report.
        let lease = Lease {
            issued_micros: 1_000_000,
            tokens: 512,
            ..Lease::default()
        };
        for row in &view {
            let target = Target {
                position: row.position,
                id: row.id,
                dp_rank: row.dp_rank,
                pair: None,
            };
            engine.on_dispatched(&lease, &target, Side::Single, 0);
        }
        let cx = Constraints::default();
        let mut scratch = Scratch::new();
        let pool = pool(n);
        assert!(engine
            .decide(&facts(1), &pool, &view, &cx, &mut scratch)
            .pick()
            .is_some());
        let (decision, allocations) =
            allocations_during(|| engine.decide(&facts(2), &pool, &view, &cx, &mut scratch));
        assert!(decision.pick().is_some());
        assert_eq!(allocations, 0, "{picker}");
    }
}

/// A decision asked by rank allocates nothing after the thread's first of
/// its size, whatever that first was: by rank where no row holds the
/// request's prefix, a draw for a request that names rows, or the whole
/// pass over a pool not said to hold a report on every row. The second
/// decision puts the holders in front: one row in eight, each running one
/// request at a rate a token a second from the next, with a long queue on
/// one row behind them. Its targets are the four fastest holders in their
/// order, which shows the front: the whole pass spreads the load over that
/// queue, reads the holders alike and draws among them. What the decision
/// keeps of the rows in front stands in buffers every first decision
/// sized: their sums in the stage's, their list in the soft filter's
/// backup. The preset is read with its soft filters and without them. A
/// whole pass by the profile's one test of a row asks for every buffer's
/// room, as a draw does: the decision by rank behind it allocates
/// nothing, with a soft filter or without.
#[test]
fn a_decision_by_rank_after_the_first_allocates_nothing() {
    let mut bare = PolicySpec::preset("cache_aware").expect("a preset");
    for profile in bare.profiles.values_mut() {
        profile
            .filter
            .retain(|entry| !matches!(entry.name(), "not-tried" | "has-room"));
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
        let names = [AffinityHolder {
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
                    named: Some(AffinityRows {
                        holders: &names,
                        ..AffinityRows::default()
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
            reported: true,
            ..pool(n)
        };
        let unsaid = pool(n);
        let cx = Constraints::default();
        for (order, (first, reported)) in firsts.iter().enumerate() {
            let mut scratch = Scratch::new();
            let of_first = if *reported { &said } else { &unsaid };
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
            assert_eq!(
                allocations, 0,
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
fn a_decision_over_a_doubted_pool_allocates_nothing_after_the_first() {
    // Every ready row doubted: the pass that lifts the liveness veto reads
    // the rows with no doubt from one buffer of the scratch. The first such
    // decision sizes it and the next finds it sized.
    let engine = engine_of("least_load");
    let mut view = fleet(64);
    for row in &mut view {
        row.health.stalled = true;
    }
    let cx = Constraints::default();
    let mut scratch = Scratch::new();
    engine
        .decide(&facts(1), &pool(64), &view, &cx, &mut scratch)
        .pick()
        .expect("the doubted fleet routes");
    let (decision, allocations) =
        allocations_during(|| engine.decide(&facts(2), &pool(64), &view, &cx, &mut scratch));
    assert!(decision.pick().is_some());
    assert_eq!(
        allocations, 0,
        "the second decision over a doubted pool allocated"
    );
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
            table: smg_policy_controller::engine::PairTable::new(&ends, &classes),
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
        store.publish_now();
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

/// A lease's life at the store, over the store's own calls and no engine:
/// its record filed, its dispatch and its completion handed to its file as
/// a session hands a gateway's frames, the gateway's completion written on
/// the served worker for the health rule, and the sweep's pass a host's
/// loop makes. The hand and the health write are the calls a served
/// controller makes for a `Dispatched` and a `Completed` frame. The record
/// is filed by the trait's door, under an id of the case's own: a served
/// decision's issue mints its id and steps the deadline queue inside the
/// crate, and no case out here reaches that call. Behind the trait's door
/// the sweep steps the queue, so the case's clock moves a step a lease and
/// the queue holds the places of one deadline's span.
#[test]
fn a_lease_filed_dispatched_and_completed_at_the_store_allocates_nothing() {
    const LEASES: u32 = 256;
    const STEP: Duration = Duration::from_millis(50);
    const SPAN: Duration = Duration::from_secs(5);
    for n in [8usize, 64] {
        let (store, ids, opened) = store_of(n);
        let gateway = GatewayId(7);
        let hand = Hand::new(gateway);
        let term = LeasesStore::term(&store);
        let life = |request: u32| {
            let now = opened + STEP * request;
            let worker = ids[request as usize % n];
            // An id of the host's own minting: the high bit.
            let lease = LeaseId(1 << 63 | u64::from(request));
            let filed = store.issue(LeaseRecord {
                lease: Lease {
                    id: lease,
                    ..Lease::default()
                },
                gateway,
                request: u64::from(request),
                target: worker,
                picked: PickRecord::default(),
                decode: None,
                untold: false,
                term,
                issued_at: now,
                dispatch_deadline: now + SPAN,
                state: LeaseState::ISSUED,
            });
            assert_eq!(filed, Ok(()), "the store files the lease");
            let dispatch = Dispatch {
                target_index: 0,
                side: Side::Single,
                since_decision_micros: 100,
            };
            store.hand(&hand, lease, Said::Dispatched(dispatch, now));
            let outcome =
                Outcome::reported(200, TransportClass::Ok).with_served(Side::Single, worker);
            store.hand(&hand, lease, Said::Completed(outcome));
            HealthSink::completed(&store, gateway, worker, outcome.class, now);
            let ended = store.expire(now);
            assert!(ended.is_empty(), "no lease waited out its deadline");
        };
        // The first lives size the file, fill the deadline queue to its
        // span and write each worker's first completion of this gateway.
        for request in 0..LEASES {
            life(request);
        }
        assert_eq!(store.live(gateway), 0, "a completed lease left the file");
        let ((), allocations) = allocations_during(|| {
            for request in LEASES..2 * LEASES {
                life(request);
            }
        });
        assert_eq!(store.live(gateway), 0, "a completed lease left the file");
        let counted = opened + STEP * LEASES;
        assert!(
            ids.iter()
                .all(|worker| store.completed_ok_since(gateway, *worker, counted)),
            "{n} members: each worker holds a completion of the counted lives"
        );
        assert_eq!(allocations, 0, "{n} members: a lease's life allocated");
    }
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
            store.publish_now();
        }
        // The fewest allocation calls of one write and the build it owes
        // among a few: the store's change log grows by doubling, and one
        // write in many pays for it.
        let fewest = |runtimes: [&str; 2]| {
            let one = |turn: usize| {
                let facts = facts_of(n / 2, runtimes[turn % 2]);
                allocations_during(|| {
                    store.upsert(facts);
                    store.publish_now();
                })
                .1
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
        // The fewest allocation calls of one registration and the build it
        // owes among a few: the store's tables and the writer's kept lists
        // grow by doubling, and one write in many pays for it.
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
            allocations_during(|| {
                store.upsert(facts);
                store.publish_now();
            })
            .1
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

/// The pin of a host in process: the engine over the host's own worker
/// list, asked through the one call a host makes for a request.
#[cfg(feature = "in-process")]
mod in_process {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use smg_policy_controller::{
        document::{AppliedSpec, PolicySpec, ScoreSpec},
        engine::{
            Candidate, Decision, LentList, ListCell, PairAdmission, PairLists, PartnerClasses,
            PlaceRequest, PolicyEngine, RequestFacts, Scorer, Target, TieBand, WorkerLists,
        },
        stages::{Lifecycle, StageRegistry},
        tracking::{
            EventKind, EventSet, Lease, Outcome, PlacementSide, ReleaseReason, Side, TransportClass,
        },
    };
    use smg_worker::{
        identity::{basic::Worker, builder::BasicWorkerBuilder},
        test_support::{no_health_check, ready_worker},
        worker::{WorkerSpec, WorkerStatus, WorkerType},
    };

    use super::allocations_during;

    /// The model every worker of the list serves.
    const MODEL: &str = "m";

    /// How often a stage was told each word of a lease.
    #[derive(Debug, Default)]
    struct Told {
        issued: AtomicUsize,
        dispatched: AtomicUsize,
        completed: AtomicUsize,
        released: AtomicUsize,
    }

    impl Told {
        /// The four counts, in a lease's order: its issue, its dispatch, its
        /// completion and its release.
        fn read(&self) -> [usize; 4] {
            [
                &self.issued,
                &self.dispatched,
                &self.completed,
                &self.released,
            ]
            .map(|count| count.load(Ordering::Relaxed))
        }
    }

    /// A scorer that scores every worker alike and takes the words of a
    /// lease it is made to take. It counts each word and allocates nothing.
    #[derive(Debug)]
    struct Hears(Arc<Told>, EventSet);

    impl Lifecycle for Hears {
        fn hears(&self) -> EventSet {
            self.1
        }

        fn on_decision_issued(&self, _: &Lease, _: &Decision) {
            self.0.issued.fetch_add(1, Ordering::Relaxed);
        }

        fn on_dispatched(&self, _: &Lease, _: &Target, _: Side, _: u32) {
            self.0.dispatched.fetch_add(1, Ordering::Relaxed);
        }

        fn on_completed(&self, _: &Lease, _: &Outcome) {
            self.0.completed.fetch_add(1, Ordering::Relaxed);
        }

        fn on_released(&self, _: &Lease, _: ReleaseReason) {
            self.0.released.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl Scorer for Hears {
        fn name(&self) -> &'static str {
            "hears"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            out.fill(1.0);
            TieBand::NORMALIZED
        }
    }

    /// The engine of the `round_robin` preset, for the life of the process:
    /// the preset as it stands, or with a [`Hears`] that takes every word of
    /// a lease among its scorers where `told` is handed.
    fn engine_of(told: Option<&Arc<Told>>) -> &'static PolicyEngine {
        let document = PolicySpec::preset("round_robin").expect("a preset of this build");
        let hears = told.map(|told| (told, EventSet::LEASE));
        engine_under(document, "round_robin", hears)
    }

    /// The engine of `document`, for the life of the process: the document
    /// as it stands, or with a [`Hears`] among the scorers of its `profile`
    /// where `hears` hands that stage's counts and the words it takes.
    fn engine_under(
        mut document: PolicySpec,
        profile: &str,
        hears: Option<(&Arc<Told>, EventSet)>,
    ) -> &'static PolicyEngine {
        let mut stages = StageRegistry::builtin();
        if let Some((told, takes)) = hears {
            let kept = Arc::clone(told);
            let added = stages.scorer("hears", move || Box::new(Hears(Arc::clone(&kept), takes)));
            added.expect("a free name");
            let profile = document.profiles.get_mut(profile);
            let scorers = &mut profile.expect("the document's profile").score;
            scorers.push(ScoreSpec::from("hears"));
        }
        let spec = AppliedSpec::validate_in(&document, &stages, &[]);
        let spec = spec.expect("the stages hold the scorer");
        Box::leak(Box::new(PolicyEngine::new(spec)))
    }

    /// A request a host in process holds from its decision to its completion
    /// allocates nothing once the list's first requests have sized the
    /// thread's scratch and the list's records: the one call that places
    /// it, the word that it went out and the word of its end. Under the
    /// preset alone a place answers the request, and nothing is owed of it.
    /// Under a stage that takes every word of a lease the pick is leased:
    /// the decision with its lease, the take, the dispatch and the
    /// completion, each word told where it is said.
    #[test]
    fn a_request_held_from_its_decision_to_its_completion_allocates_nothing() {
        for leased in [false, true] {
            for n in [8usize, 64] {
                let told = Arc::new(Told::default());
                let engine = engine_of(leased.then_some(&told));
                let list: Arc<[Arc<dyn Worker>]> = (0..n)
                    .map(|at| ready_worker(&format!("http://w{at}:8000"), MODEL))
                    .collect();
                let cell = ListCell::default();
                let lent = LentList {
                    workers: &list,
                    cell: &cell,
                    facts: None,
                };
                let request = PlaceRequest::of(MODEL, WorkerLists::Single(lent));
                let serve = || {
                    let mut placement = engine.place(&request).expect("a worker of the list");
                    assert_eq!(placement.owes_events(), leased);
                    placement.sent(Side::Single);
                    placement.ended(Side::Single, Outcome::reported(200, TransportClass::Ok));
                };
                // A rotation's first round names every worker once.
                (0..2 * n).for_each(|_| serve());
                let before = told.read();
                let ((), allocations) = allocations_during(|| (0..256).for_each(|_| serve()));
                assert_eq!(
                    allocations, 0,
                    "{n} workers, leased {leased}: a held request allocated"
                );
                // What the stage was told of the 256 requests: each one's
                // issue, dispatch and completion, and no release.
                let (after, each) = (told.read(), if leased { 256 } else { 0 });
                let since = [0, 1, 2, 3].map(|word| after[word] - before[word]);
                assert_eq!(since, [each, each, each, 0], "{n} workers");
            }
        }
    }

    /// A ready worker of `worker_type` at `url`, with no health checker
    /// behind it.
    fn typed(url: &str, worker_type: WorkerType) -> Arc<dyn Worker> {
        let spec = WorkerSpec {
            worker_type,
            ..WorkerSpec::new(url.to_owned())
        };
        let worker = <BasicWorkerBuilder>::from_spec(spec)
            .health_config(no_health_check())
            .build();
        worker.set_status(WorkerStatus::Ready);
        Arc::new(worker)
    }

    /// A pair's host that admits every worker.
    #[derive(Debug)]
    struct Open;

    impl PairAdmission for Open {
        fn has_capacity(&self, _: usize) -> bool {
            true
        }

        fn has_room(&self, _: usize) -> bool {
            true
        }
    }

    /// What the sides a host keeps with its sends allocate, counted from
    /// the split of a kept placement to the last drop: each side's send, a
    /// handle on its end, a report through the handle, and both drops.
    /// Under a stage that takes a dispatch and no end, a single request's
    /// side allocates nothing and a pair's two sides the one value they
    /// share: a handle is empty, and its report is told to nobody. Under a
    /// stage that takes an end each side allocates once more, for what it
    /// shares with its handles, and the stage hears each reported end.
    #[test]
    fn a_side_allocates_nothing_for_an_end_no_stage_takes() {
        let no_end = EventSet::NONE.with(EventKind::Dispatched);
        let report = |side: &PlacementSide, empty: bool| {
            let end = side.side_end();
            assert_eq!(end.is_empty(), empty);
            end.reported(Outcome::reported(200, TransportClass::Ok));
        };
        for (takes, a_side) in [(no_end, 0), (EventSet::LEASE, 1)] {
            let empty = a_side == 0;
            let told = Arc::new(Told::default());
            let document = PolicySpec::preset("round_robin").expect("a preset of this build");
            let engine = engine_under(document, "round_robin", Some((&told, takes)));
            let list: Arc<[Arc<dyn Worker>]> = (0..8)
                .map(|at| ready_worker(&format!("http://w{at}:8000"), MODEL))
                .collect();
            let cell = ListCell::default();
            let lent = LentList {
                workers: &list,
                cell: &cell,
                facts: None,
            };
            let request = PlaceRequest::of(MODEL, WorkerLists::Single(lent));
            let serve = || {
                let placement = engine.place(&request).expect("a worker of the list");
                assert!(placement.owes_events(), "a stage takes its dispatch");
                let placement = placement.kept();
                let ((), allocations) = allocations_during(|| {
                    let (mut side, none) = placement.sides();
                    assert!(none.is_none());
                    side.sent();
                    report(&side, empty);
                });
                allocations
            };
            for _ in 0..16 {
                serve();
            }
            let before = told.read();
            let allocations: usize = (0..256).map(|_| serve()).sum();
            assert_eq!(allocations, 256 * a_side, "a single request's side");
            let after = told.read();
            let since = [0, 1, 2, 3].map(|word| after[word] - before[word]);
            let each = 256 * a_side;
            assert_eq!(since, [each, 256, each, 0], "the words the stage takes");

            let told = Arc::new(Told::default());
            let document = PolicySpec::pair_presets("round_robin", "round_robin");
            let document = document.expect("the pair presets");
            let engine = engine_under(document, "decode", Some((&told, takes)));
            let prefills: Arc<[Arc<dyn Worker>]> =
                Arc::from([typed("grpc://prefill:9000", WorkerType::Prefill)]);
            let decodes: Arc<[Arc<dyn Worker>]> =
                Arc::from([typed("grpc://decode:9000", WorkerType::Decode)]);
            let (prefill_cell, decode_cell) = (ListCell::default(), ListCell::default());
            let classes = PartnerClasses::number(1, [&[0_u32][..]]);
            let lists = PairLists {
                prefill: LentList {
                    workers: &prefills,
                    cell: &prefill_cell,
                    facts: None,
                },
                decode: LentList {
                    workers: &decodes,
                    cell: &decode_cell,
                    facts: None,
                },
                classes: &classes,
                admission: &Open,
                prefill_places: None,
                decode_places: None,
                prefill_stalled: &[],
                prefill_held: &[],
                decode_stalled: &[],
                decode_held: &[],
            };
            let request = PlaceRequest::of(MODEL, WorkerLists::Pair(lists));
            let serve = || {
                let placement = engine.place(&request).expect("a pair of the lists");
                let placement = placement.kept();
                let ((), allocations) = allocations_during(|| {
                    let (mut prefill, decode) = placement.sides();
                    let mut decode = decode.expect("a pair's decode side");
                    prefill.sent();
                    decode.sent();
                    report(&prefill, empty);
                    report(&decode, empty);
                });
                allocations
            };
            for _ in 0..16 {
                serve();
            }
            let allocations: usize = (0..256).map(|_| serve()).sum();
            assert_eq!(allocations, 256 * (1 + 2 * a_side), "a pair's two sides");
        }
    }
}
