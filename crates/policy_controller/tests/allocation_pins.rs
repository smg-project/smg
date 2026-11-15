//! What a decision allocates, pinned at zero: once a first decision has sized
//! the per-thread scratch, a decision of every preset over a reporting fleet
//! of 64 and of 512 rows, a decision over rows with a credit held in the
//! least-load tally, and a pair decision allocate nothing (design 4.4: a
//! scorer or picker allocates nothing per decision; the host keeps one scratch
//! per thread so the framework allocates nothing after the first). A separate
//! executable, so it can declare the counting allocator the pins need.

#![expect(clippy::expect_used, reason = "tests that stop on a broken setup")]

mod common;

use std::time::{Duration, Instant};

use common::counting_alloc::{calls, Counting};
use smg_policy_controller::{
    store::{
        Ask, GatewayId, LoadSink, LoadSource, MemoryStore, ViewBuffers, WorkerFacts, WorkerStatus,
        WorkersStore, PUBLISH_WINDOW,
    },
    Candidate, Constraints, Engine, Health, Lease, Lifecycle, LoadRow, PairPools, PolicySpec,
    PoolFacts, RequestFacts, Role, Scratch, Side, SignalSource, Target, WorkerId, PRESETS,
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
