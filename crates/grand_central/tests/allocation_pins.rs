//! What a decision allocates, pinned at zero: once a first decision has sized
//! the per-thread scratch, a decision of every preset over a reporting fleet
//! of 64 and of 512 rows, a least-cost decision over mixed reports and sends, a
//! decision over rows with a credit held in the least-load tally, a decision
//! over a pool doubted on every row, and a pair decision allocate nothing
//! (design 4.4: a scorer or picker allocates nothing per decision; the host
//! keeps one scratch per thread so the framework allocates nothing after the
//! first). The
//! store's pin stands beside them: a registration into 512 workers is held
//! to one into 8. A lease's life at the store
//! allocates nothing once its file is sized: its record filed, its
//! dispatch and its completion handed to the file,
//! and the sweep's pass. A host in process is pinned
//! too, in a build
//! with the `in-process` feature: a request held from its decision to its
//! completion allocates nothing after the first ones. A keeper of worker
//! objects is pinned in a build with the `worker-host` feature: a status
//! tell copies each id of its worker's models once, and only for a store
//! that tells a journal. A read of a row copies no model card. A separate
//! executable, so it can declare the counting allocator the pins need.
//!
//! One pin reads frees: a decision that spans a write frees nothing on its
//! thread. The allocator here counts each thread's frees for it.
//!
//! One pin is of the token index: its fused walk of one page allocates one
//! list. It stands here because this target declares the counting allocator.

#![expect(clippy::expect_used, reason = "tests that stop on a broken setup")]

mod common;

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    time::{Duration, Instant},
};

use common::counting_alloc::{calls, Counting};
use smg_grand_central::{
    document::{AppliedSpec, PolicySpec, PRESETS},
    engine::{
        AffinityFacts, AffinityHolder, AffinityRows, Candidate, Constraints, Engine, LoadRow,
        PairPools, PoolFacts, RequestFacts, Scratch, SignalSource, Target, WorkerId,
    },
    stages::Lifecycle,
    store::{
        identity, GatewayId, Hand, HealthSink, LeaseRecord, LeaseState, LeasesStore,
        MembershipChange, MemoryStore, PickRecord, Position, Registrant, WorkerFacts, WorkerRow,
        WorkerStatus, WorkersStore,
    },
    tracking::{Dispatch, Lease, LeaseId, Outcome, Said, Side, TransportClass},
};

thread_local! {
    /// Frees this thread made. A constant without a destructor, so the
    /// allocator reaches it without one.
    static FREES: Cell<usize> = const { Cell::new(0) };
}

/// Frees the calling thread has made so far.
fn frees() -> usize {
    FREES.with(Cell::get)
}

/// The pins' counting allocator, which counts each thread's frees too. The
/// count of frees is this executable's own. A bench registers [`Counting`]
/// itself, and a count there would stand in every free the bench measures.
struct CountingFrees(Counting);

#[expect(
    unsafe_code,
    reason = "a counting global allocator is the only way to see what a decision freed"
)]
unsafe impl GlobalAlloc for CountingFrees {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the same layout contract as the caller's, forwarded to the allocator counted over.
        unsafe { self.0.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the same layout contract as the caller's.
        unsafe { self.0.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        FREES.with(|frees| frees.set(frees.get() + 1));
        // SAFETY: `ptr` came from `alloc` with this layout.
        unsafe { self.0.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr` came from `alloc` with this layout; the caller upholds `new_size`.
        unsafe { self.0.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingFrees = CountingFrees(Counting(System));

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
        .map(|i| {
            let mut row = Candidate::default();
            row.position = i;
            row.id = WorkerId(1_000 + u64::from(i));
            row.health.healthy = i % 6 != 5;
            row.in_flight = i % 7;
            row.load = Some({
                let mut load = LoadRow::default();
                load.running = i % 5;
                load.queued = i % 3;
                load.kv_usage_permille = (i * 37 % 900) as u16;
                load.live_rate = 1_000 + i * 13 % 3_000;
                load.report_age_ms = 5;
                load.source = SignalSource::Events;
                load
            });
            row.load_sampled_at_micros = 990_000 + u64::from(i % 9) * 1_000;
            row
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
    let mut facts = RequestFacts::default();
    facts.model = "m";
    facts.prompt_tokens = 512;
    facts.seed = seed;
    facts.now_micros = 1_000_000;
    facts
}

fn pool(n: u32) -> PoolFacts<'static> {
    let mut pool = PoolFacts::default();
    pool.pool_size = n;
    pool.generation = 1;
    pool
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
    let mut lease = Lease::default();
    lease.issued_micros = 1_000_000;
    lease.tokens = 512;
    for row in &view {
        let mut target = Target::default();
        target.position = row.position;
        target.id = row.id;
        target.dp_rank = row.dp_rank;
        target.pair = None;
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
        let mut lease = Lease::default();
        lease.issued_micros = 1_000_000;
        lease.tokens = 512;
        for row in &view {
            let mut target = Target::default();
            target.position = row.position;
            target.id = row.id;
            target.dp_rank = row.dp_rank;
            target.pair = None;
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
                .map(|i| {
                    let mut affinity = AffinityFacts::default();
                    affinity.overlap_blocks = if i % 8 == 0 { blocks } else { 0 };
                    affinity
                })
                .collect()
        };
        let (cold, held) = (column(0), column(4));
        let mut holder = AffinityHolder::default();
        holder.row = 0;
        holder.overlap_blocks = 4;
        holder.predicted_blocks = 0;
        let names = [holder];
        // Each first, and whether the host says the pool's fact of it.
        let firsts = [
            (
                {
                    let mut request_facts = facts(1);
                    request_facts.affinity = &cold;
                    request_facts
                },
                true,
            ),
            (
                {
                    let mut request_facts = facts(1);
                    request_facts.named = Some({
                        let mut affinity_rows = AffinityRows::default();
                        affinity_rows.holders = &names;
                        affinity_rows
                    });
                    request_facts
                },
                true,
            ),
            (
                {
                    let mut request_facts = facts(1);
                    request_facts.affinity = &cold;
                    request_facts
                },
                false,
            ),
        ];
        let mut said = pool(n);
        said.reported = true;
        let unsaid = pool(n);
        let cx = Constraints::default();
        for (order, (first, reported)) in firsts.iter().enumerate() {
            let mut scratch = Scratch::new();
            let of_first = if *reported { &said } else { &unsaid };
            engine
                .decide(first, of_first, &view, &cx, &mut scratch)
                .pick()
                .expect("the fleet routes");
            let mut second = facts(2);
            second.affinity = &held;
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
    let mut first = facts(1);
    first.tried = &tried;
    let mut next = facts(2);
    next.tried = &tried;
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
            let mut f = facts(seed);
            f.tried = &tried;
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
    let mut lease = Lease::default();
    lease.issued_micros = 1_000_000;
    lease.tokens = 512;
    for c in &view {
        let mut target = Target::default();
        target.position = c.position;
        target.id = c.id;
        target.dp_rank = c.dp_rank;
        target.pair = None;
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
        let mut pools = PairPools::default();
        pools.prefills = &prefills;
        pools.decodes = decodes;
        pools.table = smg_grand_central::engine::PairTable::new(&ends, &classes);
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
    let mut pools = PairPools::default();
    pools.prefills = &prefills;
    pools.decodes = &decodes;
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

/// The row of `facts` as a registration by a control call leaves it. The
/// stores here have no holder of worker objects: each takes a row as a
/// decided change ([`MemoryStore::follow`]), and the test counts the
/// epoch the write steps to.
fn row_of(facts: WorkerFacts) -> WorkerRow {
    WorkerRow {
        id: identity(&facts.key, facts.incarnation),
        facts,
        status: WorkerStatus::Pending,
        revision: 1,
        admitted_at: Instant::now(),
        maker: Registrant::Control,
    }
}

/// A store of two pools: `n` ready members serving `m` and one member
/// serving another model, each published. The store and the members of
/// `m`.
fn store_of(n: usize) -> (MemoryStore, Vec<WorkerId>) {
    let store = MemoryStore::new();
    let register = |key: String, model: &str| {
        let row = row_of(WorkerFacts {
            key: key.clone(),
            engine: key,
            incarnation: 1,
            models: vec![model.to_string()],
            dp_size: 1,
            ..WorkerFacts::default()
        });
        let id = row.id;
        let epoch = WorkersStore::epoch(&store) + 1;
        let registered = MembershipChange::Registered { id };
        store.follow(Position::default(), epoch, registered, Some(row));
        store.publish_now();
        let ready = MembershipChange::StatusChanged {
            id,
            from: WorkerStatus::Pending,
            to: WorkerStatus::Ready,
        };
        store.follow(Position::default(), epoch + 1, ready, None);
        let landed = WorkersStore::epoch(&store);
        assert_eq!(landed, epoch + 1, "the member's two writes landed");
        id
    };
    let ids = (0..n)
        .map(|index| register(format!("http://w{index}:8000"), "m"))
        .collect();
    register("http://other:8000".to_string(), "other");
    (store, ids)
}

/// A lease's life at the store, over the store's own calls and no engine:
/// its record filed, its dispatch and its completion handed to its file as
/// a session hands a gateway's frames, and the sweep's pass a host's
/// loop makes. The hand is the call a served controller makes for a
/// `Dispatched` and for a `Completed` frame. The record
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
        let (store, ids) = store_of(n);
        let began = Instant::now();
        let gateway = GatewayId(7);
        let hand = Hand::new(gateway);
        let term = LeasesStore::term(&store);
        let life = |request: u32| {
            let now = began + STEP * request;
            let worker = ids[request as usize % n];
            // An id of the host's own minting: the high bit.
            let lease = LeaseId(1 << 63 | u64::from(request));
            let filed = store.issue(LeaseRecord {
                lease: {
                    let mut held_lease = Lease::default();
                    held_lease.id = lease;
                    held_lease
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
            let mut dispatch = Dispatch::default();
            dispatch.target_index = 0;
            dispatch.side = Side::Single;
            dispatch.since_decision_micros = 100;
            store.hand(&hand, lease, Said::Dispatched(dispatch, now));
            let outcome =
                Outcome::reported(200, TransportClass::Ok).with_served(Side::Single, worker);
            store.hand(&hand, lease, Said::Completed(outcome));
            let ended = store.expire(now);
            assert!(ended.is_empty(), "no lease waited out its deadline");
        };
        // The first lives size the file and fill the deadline queue to its
        // span.
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
        assert_eq!(allocations, 0, "{n} members: a lease's life allocated");
    }
}

#[test]
fn a_registrations_allocations_do_not_follow_the_fleet() {
    // A membership write walks every member once. Where no veto stands,
    // each list that walk fills takes the members' count at once or is
    // kept from the last build, so what a registration allocates does not
    // follow the fleet: one into 512 workers makes the allocator calls of
    // one into 8.
    let fewest = |n: usize| {
        let (store, _) = store_of(n);
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
            // The row is stated ahead of the count: the count holds the
            // store's write and the build it owes.
            let row = row_of(facts);
            let epoch = WorkersStore::epoch(&store) + 1;
            let registered = MembershipChange::Registered { id: row.id };
            let counted = allocations_during(|| {
                store.follow(Position::default(), epoch, registered, Some(row));
                store.publish_now();
            });
            let landed = WorkersStore::epoch(&store);
            assert_eq!(landed, epoch, "the registration landed");
            counted.1
        };
        (0..8).map(one).min()
    };
    let (few, many) = (fewest(8), fewest(512));
    assert_eq!(many, few, "a registration's allocations follow the fleet");
}

/// A generation of the round-robin preset under `number`.
fn generation(number: u64) -> AppliedSpec {
    let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
    document.generation = number;
    AppliedSpec::validate(&document).expect("a preset validates")
}

/// A decision that spans a write frees nothing on its thread. The decision
/// borrows the world of its instant ([`MemoryStore::deciding`]). A write
/// that publishes meanwhile replaces that world, so the decision is its
/// last reader. The writer keeps a replaced world a reader holds until a
/// later publication finds nobody on it. So the decision's end frees
/// neither the world, its vetoes, its derived tables nor its generation.
#[test]
fn a_decision_that_spans_a_write_frees_nothing_on_its_thread() {
    let (store, ids) = store_of(64);
    let gateway = GatewayId(7);
    let until = Some(Instant::now() + Duration::from_secs(7_200));
    let member = |key: String, incarnation: u64| WorkerFacts {
        key: key.clone(),
        engine: key,
        incarnation,
        models: vec!["m".to_string()],
        dp_size: 1,
        ..WorkerFacts::default()
    };
    // Each membership write is a decided change the store takes
    // ([`MemoryStore::follow`]), at the epoch it steps to.
    let follow = |change: MembershipChange, row: Option<WorkerRow>| {
        let epoch = WorkersStore::epoch(&store) + 1;
        store.follow(Position::default(), epoch, change, row);
        let landed = WorkersStore::epoch(&store);
        assert_eq!(landed, epoch, "the store took {change:?}");
    };
    let write = |kind: usize, turn: usize| match kind {
        // A veto of the asking gateway.
        0 => store.gateway_veto(gateway, ids[10 + turn], until),
        // A membership write, and the build it owes.
        1 => {
            let row = row_of(member(format!("http://new{turn}:8000"), 1));
            follow(MembershipChange::Registered { id: row.id }, Some(row));
            store.publish_now();
        }
        // A generation applied.
        2 => store
            .apply(generation(turn as u64 + 1))
            .expect("above the standing one"),
        // A leave: it pays its build at its answer.
        3 => follow(MembershipChange::Removed { id: ids[40 + turn] }, None),
        // A new incarnation: the write pays its build at its answer.
        _ => {
            let row = row_of(member(format!("http://w{}:8000", 20 + turn), 2));
            let replaced = MembershipChange::Replaced {
                old: ids[20 + turn],
                new: row.id,
            };
            follow(replaced, Some(row));
        }
    };
    for kind in 0..5 {
        for turn in 0..3 {
            // The decision holds its world across the write: it is the last
            // reader of the world the write replaces.
            let freed = store.deciding(|world| {
                write(kind, turn);
                let standing = store.snapshot();
                assert!(!std::ptr::eq(world, &*standing), "write {kind} published");
                drop(standing);
                frees()
            });
            assert_eq!(
                frees() - freed,
                0,
                "a decision that spanned write {kind} freed on its thread"
            );
        }
    }
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

/// The fused walk of the token tree keeps its path in one list: a prompt of
/// one page that matches nothing in an empty tree allocates that list and
/// nothing else.
#[test]
fn a_fused_walk_of_one_page_allocates_one_list_for_its_path() {
    let tree = kv_index::TokenTree::new();
    // One page of the tree, whatever its page size.
    let page = vec![7u32; tree.page_size()];
    let walk = || allocations_during(|| tree.match_and_insert_with(&page, |_| None)).1;
    // The first call interns the tenant an empty tree answers with: unmeasured.
    let _ = walk();
    assert_eq!(walk(), 1);
}

/// The pin of a host in process: the engine over the host's own worker
/// list, asked through the one call a host makes for a request.
#[cfg(feature = "in-process")]
mod in_process {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use smg_grand_central::{
        document::{AppliedSpec, PolicySpec, ScoreSpec},
        engine::{
            Candidate, Decision, InProcess, LentList, ListCell, PairAdmission, PairLists,
            PartnerClasses, PlaceRequest, RequestFacts, Scorer, Target, TieBand, WorkerLists,
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
    fn engine_of(told: Option<&Arc<Told>>) -> &'static InProcess {
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
    ) -> &'static InProcess {
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
        Box::leak(Box::new(InProcess::new(spec)))
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

/// What a status tell copies of its worker's models, through a host's
/// write call on the keeper of its worker objects.
#[cfg(feature = "worker-host")]
mod status_tells {
    use std::sync::Arc;

    use smg_grand_central::{
        service::replication::Log,
        store::{MemoryStore, Registrant, WorkerFacts, WorkerStatus, WorkersStore},
        worker_host::{Members, Objects},
    };
    use smg_worker::registry::WorkerRegistry;

    use super::allocations_during;

    /// The fewest allocation calls of one connect signal among a few rows
    /// that each serve `models` models, over a store that tells a log or
    /// tells none. The fewest, as the pin of a registration keeps it: a
    /// tell that paid for a table's growth would not move the count. No
    /// tell here pays for one: the log's ring is reserved whole where the
    /// log is built, and the registry makes a worker's lock at its
    /// registration.
    fn a_connect_signal(journaled: bool, models: usize) -> usize {
        let store = if journaled {
            MemoryStore::journaled(Arc::new(Log::new(64)))
        } else {
            MemoryStore::new()
        };
        let objects: Objects<Members> = Objects::new(Arc::new(WorkerRegistry::new()));
        let one = |turn: usize| {
            let key = format!("http://w{turn}:8000");
            let facts = WorkerFacts {
                key: key.clone(),
                engine: key,
                incarnation: 1,
                models: (0..models).map(|at| format!("m{at}")).collect(),
                dp_size: 1,
                ..WorkerFacts::default()
            };
            let laid = objects.upsert(&store, facts, Registrant::Control);
            let id = laid.expect("a registration").id();
            let row = WorkersStore::get(&store, id).expect("the row");
            let signal = || objects.connect_row(&store, id, row.revision);
            let ((), allocations) = allocations_during(signal);
            let stands = WorkersStore::get(&store, id).expect("the row").status;
            assert_eq!(stands, WorkerStatus::Ready, "the signal readied the row");
            allocations
        };
        (0..8).map(one).min().expect("eight rows")
    }

    /// A status tell copies each id of its worker's models once where its
    /// store tells a journal: the list its entry holds, built by the
    /// keeper. Neither the store nor the log copies an id again. A store
    /// that tells no journal is handed no list and none is built: its tell
    /// allocates the same whatever its worker serves.
    #[test]
    fn a_status_tell_copies_its_models_once_and_for_a_journal_alone() {
        const MODELS: usize = 4;
        for journaled in [false, true] {
            let one = a_connect_signal(journaled, 1);
            let many = a_connect_signal(journaled, MODELS);
            let more = if journaled { MODELS - 1 } else { 0 };
            assert_eq!(
                many,
                one + more,
                "a journal {journaled}: {one} for one model, {many} for {MODELS}"
            );
        }
    }
}

/// What a read of a row copies of its worker's models, through a host's
/// read call on the keeper of its worker objects.
#[cfg(feature = "worker-host")]
mod row_reads {
    use std::sync::Arc;

    use smg_grand_central::{
        store::{MemoryStore, Registrant, WorkerFacts},
        worker_host::{Members, Objects},
    };
    use smg_worker::{model_card::ModelCard, registry::WorkerRegistry};

    use super::allocations_during;

    /// The fewest allocation calls of one read of a row among a few reads.
    /// The row's worker serves two models. Each card is as the keeper
    /// builds it from the row, an id alone. Where `filled`, the test sets
    /// cards that hold more beside their ids. Each holds a display name,
    /// two aliases, a family, an architecture, a tokenizer path, a chat
    /// template of 4 KiB and two parsers' names.
    fn a_row_read(filled: bool) -> usize {
        let store = MemoryStore::new();
        let objects: Objects<Members> = Objects::new(Arc::new(WorkerRegistry::new()));
        let key = "http://w:8000".to_owned();
        let facts = WorkerFacts {
            key: key.clone(),
            engine: key,
            incarnation: 1,
            models: vec!["m0".to_owned(), "m1".to_owned()],
            dp_size: 1,
            ..WorkerFacts::default()
        };
        let laid = objects.upsert(&store, facts, Registrant::Control);
        let id = laid.expect("a registration").id();
        if filled {
            let object = objects.object(id).expect("the row's object");
            let worker = objects.registry().get(&object).expect("the object");
            let card = |model: &str| {
                ModelCard::new(model)
                    .with_display_name("M")
                    .with_aliases(["m-a", "m-b"])
                    .with_hf_model_type("llama")
                    .with_architectures(vec!["LlamaForCausalLM".to_owned()])
                    .with_tokenizer_path(model)
                    .with_chat_template("x".repeat(4096))
                    .with_reasoning_parser("r")
                    .with_tool_parser("t")
            };
            let cards = vec![card("m0"), card("m1")];
            let changed = objects.registry().refresh_models(&worker, cards);
            assert!(changed, "the object took the filled cards");
        }
        let one = |_| {
            let (row, allocations) = allocations_during(|| objects.read_row(id));
            assert_eq!(row.expect("the row").facts.models, ["m0", "m1"]);
            allocations
        };
        (0..4).map(one).min().expect("four reads")
    }

    /// A read of a row copies each id of its worker's models and no card.
    /// It allocates the same for cards that hold their ids alone and for
    /// cards that hold what this case fills beside them.
    #[test]
    fn a_row_read_copies_a_models_id_and_no_card() {
        let (built, filled) = (a_row_read(false), a_row_read(true));
        assert_eq!(filled, built, "ids alone against filled cards");
    }
}
