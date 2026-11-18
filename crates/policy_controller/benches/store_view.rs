//! The cost of a decision's view built from the store, at 8, 64 and 512
//! members of one pool, and of the writes that feed it:
//!
//! - `policy_store/view_build/<n>`: one decision's build: the pool's
//!   published rows copied, the position and the report's age written;
//! - `policy_store/view_build_column/<n>`: the same under a profile that
//!   declares the affinity column;
//! - `policy_store/load_write/<n>`: one load record's write inside the
//!   publication window: on file and folded into the writer's rows;
//! - `policy_store/publish/<n>`: one window's publication of the pool: one
//!   copy, one pointer swap;
//! - `policy_store/floor/<n>`: the decision benches' gated build, rows
//!   written from facts in hand, in this binary, so that one job reads a
//!   build from the store beside it. No check holds a build to the floor:
//!   the gate compares each id with its own base alone, and the design's
//!   bound (a build within twice the floor) is for a reader of the two
//!   rows. A build may read under the floor;
//! - `policy_store/view_build_veto1/<n>` and `view_build_veto8/<n>`: the
//!   build with one and with eight standing vetoes of the asking gateway;
//! - `policy_store/view_build_foreign64/8`: the build in the pool of eight
//!   beside sixty-four vetoes of other gateways, none the asking one's;
//! - `policy_store/view_build_gateways8/8`, `view_build_gateways64/8` and
//!   `view_build_gateways512/8`: the build in the pool of eight with one
//!   member vetoed by the asking gateway and by eight, by sixty-four and
//!   by 512 others, one veto each: the first two reach the asking
//!   gateway's vetoes by a walk, the third by a search;
//! - `policy_store/veto_write/512` and `policy_store/member_write/512`: one
//!   veto's write and one membership write beside 512 standing vetoes of
//!   eight other gateways;
//! - `policy_store/member_write_spread/512`: one membership write beside
//!   6,144 standing vetoes of twelve other gateways, the 512 members spread
//!   over 64 pools;
//! - `policy_store/member_write_listed/512`: one membership write under a
//!   generation that declares the view's three lists: the worker written
//!   is read under them and the 512 members' fields are carried.
//!
//! `policy_store_vetoes/*` are the veto cells outside the gate: `k3/<n>`
//! with three vetoes of the asking gateway, `foreign8/<n>` with eight other
//! gateways' vetoes on one member and `foreign64/<n>` with eight gateways'
//! on each of eight members, none of them the asking one's: what a build
//! pays to pass over a veto that is not its own.
//!
//! `policy_store_held/publish/<n>` is the publication beside a reader that
//! holds the world listing the rows it would write over: the publication
//! leaves those rows whole and writes over rows the reader let go. Outside
//! the gate.
//!
//! The pair door's cells are a bench target of their own (`store_pair`):
//! this binary is built on the base branch with the pull request's bench
//! sources, so a row here calls nothing the base's store lacks.
//!
//! `policy_store_pairs/kept/<n>` and `policy_store_pairs/moved/<n>` are
//! one membership write of a fleet of pairs, `n` prefills and `n` decodes
//! of one model that name no runtime: one prefill's registration written
//! again as it stands, which moves no offer, and its runtime named and
//! unnamed in turn, which moves its offer at every write. The gate reads
//! both at 512 a side.
//!
//! `policy_store_models/member_write/<ids>` is one membership write beside
//! a worker that lists `ids` models: one small worker's registration
//! written again as it stands, in a store of eight workers of one model and
//! one worker of `ids` models of its own. The gate reads it at 1,024.
//!
//! `policy_store_cadence/*` reads the build beside a writer on another
//! core, for a run pinned to two cores (both threads spin, so each keeps
//! its core): `quiet/<n>` with the writer idle, `busy/<n>` with it writing
//! ten records per member per second through the store's own window, and
//! `after_publish/<n>`, the first build after a publication made on the
//! writer's core. Outside the `policy_store/` ids, so a filter on those
//! starts no thread. These cells run on a store of their own: the cells
//! before them move the shared store's window ahead of the host's clock,
//! which the busy writer stamps its records with. The busy cell is held to
//! what it claims: a build reads a record the writer wrote while busy.
//!
//! Every member is ready with a record on file. A timed view is written
//! into buffers on a page's start (`placed`), one placement per size for
//! the whole binary: the helper walks its earlier rejects first, and a size
//! placed a fifth time stops at its bound.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer"
)]

#[expect(
    dead_code,
    reason = "the shared fleet's engine builder belongs to the decision benches"
)]
mod common;
#[path = "common/placed.rs"]
mod placed;

use std::{
    cell::Cell,
    hint::black_box,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use placed::page_start_buffer;
use smg_policy_controller::{
    store::{
        Ask, GatewayId, HealthSink, LoadSink, LoadSource, MemoryStore, ViewBuffers, WorkerFacts,
        WorkerStatus, WorkersStore, PUBLISH_WINDOW,
    },
    AppliedSpec, LoadRow, PolicySpec, Role, SignalSource, WorkerId,
};

const SIZES: [usize; 3] = [8, 64, 512];
const MODEL: &str = "m";

/// Member `index`'s record as of write `turn`: figures that differ per
/// member and per write, so no two rows read alike.
fn record_of(index: usize, turn: u64) -> LoadRow {
    let x = (index as u64 + 1)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(turn.wrapping_mul(0xbf58_476d_1ce4_e5b9));
    LoadRow {
        running: (x >> 16) as u32 % 12,
        queued: (x >> 24) as u32 % 4,
        kv_usage_permille: ((x >> 32) % 900) as u16,
        live_rate: 1_000 + ((x >> 40) % 3_000) as u32,
        report_age_ms: 5,
        source: SignalSource::Events,
        ..LoadRow::default()
    }
}

/// One pool of `n` members in a store, and the buffers its views are timed
/// in.
struct Fixture {
    n: usize,
    store: Arc<MemoryStore>,
    ids: Vec<WorkerId>,
    /// The instant the store's last window opened; a cell that publishes
    /// moves it.
    window: Cell<Instant>,
    out: ViewBuffers,
}

/// A store with one pool of `n` members, every one registered, ready, with
/// its record on file and published: the store, its members by index and
/// the instant its last window opened.
fn pool_of(n: usize) -> (Arc<MemoryStore>, Vec<WorkerId>, Instant) {
    let store = Arc::new(MemoryStore::new());
    let opened = Instant::now();
    let ids: Vec<WorkerId> = (0..n)
        .map(|index| {
            let key = format!("http://w{index}:8000");
            let id = store
                .upsert(WorkerFacts {
                    key: key.clone(),
                    engine: key,
                    incarnation: 1,
                    models: vec![MODEL.to_string()],
                    dp_size: 1,
                    max_running: Some(4),
                    ..WorkerFacts::default()
                })
                .id();
            let revision = WorkersStore::get(&*store, id).expect("the row").revision;
            store
                .set_status(id, WorkerStatus::Ready, revision)
                .expect("ready");
            store.record(id, 0, record_of(index, 0), opened, LoadSource::Stream);
            id
        })
        .collect();
    let opened = opened + PUBLISH_WINDOW;
    store.publish(opened);
    assert_eq!(store.publish_due(), None);
    (store, ids, opened)
}

/// The decision every build is timed for: a second after the window that
/// opened at `opened`.
fn ask_at(opened: Instant, affinity: bool) -> Ask {
    Ask {
        gateway: GatewayId(1),
        now: opened + Duration::from_secs(1),
        pinned: None,
        affinity,
    }
}

impl Fixture {
    fn new(n: usize, out: ViewBuffers) -> Self {
        let (store, ids, opened) = pool_of(n);
        Self {
            n,
            store,
            ids,
            window: Cell::new(opened),
            out,
        }
    }

    fn ask(&self, affinity: bool) -> Ask {
        ask_at(self.window.get(), affinity)
    }
}

/// One build into the fixture's buffers; the rows it wrote.
fn build(store: &MemoryStore, ask: &Ask, out: &mut ViewBuffers) -> usize {
    let built = store.build_view(MODEL, Role::Single, black_box(ask), out);
    black_box(&out.candidates);
    black_box(&out.affinity);
    built.map_or(0, |built| built.pool_size as usize)
}

/// The gated build's floor in this binary: the rows written from facts in
/// hand (`policy_engine/view_build`'s own build over the same synthetic
/// fleet) into the fixture's buffer, so one run on one machine reads the
/// floor and the build from the store.
fn floor(c: &mut Criterion, fixtures: &mut [Fixture]) {
    let mut group = c.benchmark_group("policy_store/floor");
    for fixture in fixtures.iter_mut() {
        let n = fixture.n;
        let fleet = common::fleet(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                common::build_view(black_box(&fleet), &mut fixture.out.candidates);
                black_box(&fixture.out.candidates);
                fixture.out.candidates.len()
            });
        });
    }
    group.finish();
}

/// The build under standing vetoes, each ending after the decision, on
/// members spread over the pool; every cell is held to its count of masked
/// rows and the vetoes are cleared after the last. Of the asking gateway:
/// `policy_store/view_build_veto1/<n>` and `policy_store/view_build_veto8/<n>`
/// with one and eight, and `policy_store_vetoes/k3/<n>` with three. Of other
/// gateways alone, which mask nothing for the asking one:
/// `policy_store_vetoes/foreign8/<n>` (eight gateways on one member) and
/// `policy_store_vetoes/foreign64/<n>` (eight gateways on eight members),
/// which the pool of eight reads again as the gated
/// `policy_store/view_build_foreign64/8`. Of the asking gateway and of
/// others on one member, one veto each, which is every gateway vetoing one
/// sick worker: `policy_store/view_build_gateways8/8`,
/// `policy_store/view_build_gateways64/8` and
/// `policy_store/view_build_gateways512/8`, beside eight, sixty-four and
/// 512 others.
fn vetoed(c: &mut Criterion, fixtures: &mut [Fixture]) {
    for fixture in fixtures.iter_mut() {
        let n = fixture.n;
        let ask = fixture.ask(false);
        let until = ask.now + Duration::from_secs(60);
        let member = |nth: usize| fixture.ids[nth * n / 8];
        let mut cell = |c: &mut Criterion, group: &str, id: &str, ask: &Ask, masked: usize| {
            assert_eq!(build(&fixture.store, ask, &mut fixture.out), n);
            let stalled = fixture
                .out
                .candidates
                .iter()
                .filter(|row| row.health.stalled)
                .count();
            assert_eq!(stalled, masked, "{group}/{id}/{n}");
            let mut group = c.benchmark_group(group);
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new(id, n), &n, |b, _| {
                b.iter(|| build(&fixture.store, ask, &mut fixture.out));
            });
            group.finish();
        };
        let mut standing = 0;
        for (count, group, id) in [
            (1_usize, "policy_store", "view_build_veto1"),
            (3, "policy_store_vetoes", "k3"),
            (8, "policy_store", "view_build_veto8"),
        ] {
            while standing < count {
                fixture
                    .store
                    .gateway_veto(ask.gateway, member(standing), Some(until));
                standing += 1;
            }
            cell(c, group, id, &ask, count);
        }
        for nth in 0..standing {
            fixture.store.gateway_veto(ask.gateway, member(nth), None);
        }
        // Other gateways' vetoes: on one member, then on eight.
        let others = (2..10).map(GatewayId);
        for (members, id) in [(1_usize, "foreign8"), (8, "foreign64")] {
            for gateway in others.clone() {
                for nth in 0..members {
                    fixture
                        .store
                        .gateway_veto(gateway, member(nth), Some(until));
                }
            }
            cell(c, "policy_store_vetoes", id, &ask, 0);
        }
        // The gate reads the sixty-four in the pool of eight: where a build
        // that walked another gateway's entries paid most.
        if n == 8 {
            cell(c, "policy_store", "view_build_foreign64", &ask, 0);
        }
        for gateway in others {
            for nth in 0..8 {
                fixture.store.gateway_veto(gateway, member(nth), None);
            }
        }
        // One member vetoed by the asking gateway and by other gateways,
        // one veto each. The others hold the even ids and the asking
        // gateway an odd one in their middle: its run is the farthest from
        // both ends of the pool's runs, the longest way a build has to its
        // own. Read in the pool of eight, where that way is most of what a
        // veto adds to a build. Beside 8 and 64 others the way is a walk;
        // beside 512 the runs are more than a build walks, and the row
        // holds the search.
        if n == 8 {
            let sick = member(0);
            let mut holders = 0_u64;
            for (count, id) in [
                (8_u64, "view_build_gateways8"),
                (64, "view_build_gateways64"),
                (512, "view_build_gateways512"),
            ] {
                while holders < count {
                    holders += 1;
                    fixture
                        .store
                        .gateway_veto(GatewayId(2 * holders), sick, Some(until));
                }
                let middle = Ask {
                    gateway: GatewayId(count + 1),
                    ..ask
                };
                fixture
                    .store
                    .gateway_veto(middle.gateway, sick, Some(until));
                cell(c, "policy_store", id, &middle, 1);
                fixture.store.gateway_veto(middle.gateway, sick, None);
            }
            for gateway in (1..=holders).map(|holder| GatewayId(2 * holder)) {
                fixture.store.gateway_veto(gateway, sick, None);
            }
        }
        assert_eq!(build(&fixture.store, &ask, &mut fixture.out), n);
        assert!(fixture.out.candidates.iter().all(|row| !row.health.stalled));
    }
}

/// One membership write in turn: the worker joins where it is out and
/// leaves where it is in. `joined` holds its id and its row's revision
/// between the two.
fn join_or_leave(store: &MemoryStore, joining: &WorkerFacts, joined: &mut Option<(WorkerId, u64)>) {
    match joined.take() {
        Some((id, revision)) => {
            store
                .remove(id, revision)
                .expect("the joined worker leaves");
        }
        None => {
            let id = store.upsert(black_box(joining.clone())).id();
            let revision = WorkersStore::get(store, id).expect("the row").revision;
            *joined = Some((id, revision));
        }
    }
}

/// The store's rare writes beside standing vetoes, in the largest pool
/// alone: eight other gateways each hold a veto on the same 64 members, 512
/// in all. `policy_store/veto_write/<n>` is one write of the asking
/// gateway's veto on one more member, set and cleared in turn;
/// `policy_store/member_write/<n>` is one membership write, a worker
/// joining the pool and leaving it in turn. The pool is left as it was.
fn writes_beside_vetoes(c: &mut Criterion, fixtures: &mut [Fixture]) {
    let Some(fixture) = fixtures.iter_mut().max_by_key(|fixture| fixture.n) else {
        return;
    };
    let (n, store) = (fixture.n, &fixture.store);
    let ask = fixture.ask(false);
    let until = ask.now + Duration::from_secs(60);
    let others = (2..10).map(GatewayId);
    let vetoed = |nth: usize| fixture.ids[nth * n / 64];
    for gateway in others.clone() {
        for nth in 0..64 {
            store.gateway_veto(gateway, vetoed(nth), Some(until));
        }
    }
    let mut group = c.benchmark_group("policy_store");
    // A member the others' vetoes name as well: the asking gateway's entry
    // stands beside theirs on one row.
    let target = vetoed(31);
    let mut set = false;
    group.bench_with_input(BenchmarkId::new("veto_write", n), &n, |b, _| {
        b.iter(|| {
            set = !set;
            store.gateway_veto(ask.gateway, black_box(target), set.then_some(until));
        });
    });
    store.gateway_veto(ask.gateway, target, None);
    let joining = WorkerFacts {
        key: "http://joining:8000".to_string(),
        engine: "http://joining:8000".to_string(),
        incarnation: 1,
        models: vec![MODEL.to_string()],
        dp_size: 1,
        max_running: Some(4),
        ..WorkerFacts::default()
    };
    let mut joined = None;
    group.bench_with_input(BenchmarkId::new("member_write", n), &n, |b, _| {
        b.iter(|| join_or_leave(store, &joining, &mut joined));
    });
    group.finish();
    if let Some((id, revision)) = joined {
        store
            .remove(id, revision)
            .expect("the joined worker leaves");
    }
    for gateway in others {
        for nth in 0..64 {
            store.gateway_veto(gateway, vetoed(nth), None);
        }
    }
    assert_eq!(build(store, &ask, &mut fixture.out), n);
    assert!(fixture.out.candidates.iter().all(|row| !row.health.stalled));
}

/// A membership write beside vetoes spread over pools, in a store of its
/// own: 512 members, eight to a pool over 64 pools, and twelve other
/// gateways each holding a veto on every member, 6,144 in all.
/// `policy_store/member_write_spread/<n>` is one membership write, a worker
/// joining the first pool and leaving it in turn. Each is a full rebuild of
/// every pool's runs, where a gateway's list spans the pools and a run
/// holds one pool's share of it.
fn member_write_beside_spread_vetoes(c: &mut Criterion) {
    const MEMBERS: usize = 512;
    const POOLS: usize = 64;
    let store = MemoryStore::new();
    let opened = Instant::now();
    let facts = |key: String, pool: usize| WorkerFacts {
        engine: key.clone(),
        key,
        incarnation: 1,
        models: vec![format!("spread-{pool}")],
        dp_size: 1,
        max_running: Some(4),
        ..WorkerFacts::default()
    };
    let ids: Vec<WorkerId> = (0..MEMBERS)
        .map(|index| {
            let id = store
                .upsert(facts(format!("http://s{index}:8000"), index % POOLS))
                .id();
            let revision = WorkersStore::get(&store, id).expect("the row").revision;
            store
                .set_status(id, WorkerStatus::Ready, revision)
                .expect("ready");
            store.record(id, 0, record_of(index, 0), opened, LoadSource::Stream);
            id
        })
        .collect();
    let opened = opened + PUBLISH_WINDOW;
    store.publish(opened);
    assert_eq!(store.publish_due(), None);
    let until = opened + Duration::from_secs(3_600);
    for gateway in (2..14).map(GatewayId) {
        for id in &ids {
            store.gateway_veto(gateway, *id, Some(until));
        }
    }
    let joining = facts("http://joining-spread:8000".to_string(), 0);
    let mut joined = None;
    let mut group = c.benchmark_group("policy_store");
    group.bench_with_input(
        BenchmarkId::new("member_write_spread", MEMBERS),
        &MEMBERS,
        |b, _| b.iter(|| join_or_leave(&store, &joining, &mut joined)),
    );
    group.finish();
}

/// `policy_store/member_write_listed/<n>` is one membership write under a
/// generation that declares the view's three lists, in a store of its own:
/// 512 members of one pool, each stating an engine family, a zone and
/// twelve labels, under two families, four zones and eight selectors. A
/// worker joins the pool and leaves it in turn: the write reads that
/// worker under the lists and carries every other member's fields.
fn member_write_under_lists(c: &mut Criterion) {
    const MEMBERS: usize = 512;
    let store = MemoryStore::new();
    let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
    document.generation = 1;
    document.view.engines = vec!["alpha".to_string(), "beta".to_string()];
    document.view.zones = (0..4).map(|zone| format!("z{zone}")).collect();
    document.view.labels = (0..8).map(|key| format!("k{key}")).collect();
    let listing = AppliedSpec::validate(&document).expect("the document validates");
    store.apply(listing).expect("applied");
    let opened = Instant::now();
    let facts = |key: String, index: usize| WorkerFacts {
        engine: key.clone(),
        key,
        incarnation: 1,
        models: vec![MODEL.to_string()],
        dp_size: 1,
        max_running: Some(4),
        engine_kind: if index.is_multiple_of(2) {
            "alpha"
        } else {
            "beta"
        }
        .to_string(),
        zone: Some(format!("z{}", index % 5)),
        labels: (0..12)
            .map(|nth| (format!("k{}", (index + nth) % 80), "v".to_string()))
            .collect(),
        ..WorkerFacts::default()
    };
    for index in 0..MEMBERS {
        let id = store
            .upsert(facts(format!("http://l{index}:8000"), index))
            .id();
        let revision = WorkersStore::get(&store, id).expect("the row").revision;
        store
            .set_status(id, WorkerStatus::Ready, revision)
            .expect("ready");
        store.record(id, 0, record_of(index, 0), opened, LoadSource::Stream);
    }
    store.publish(opened + PUBLISH_WINDOW);
    assert_eq!(store.publish_due(), None);
    // A fifth of the members state a zone the list does not name: the
    // lists stand, and the store counts those members outside.
    assert!(store.outside_lists().zones > 0);
    let joining = facts("http://joining-listed:8000".to_string(), MEMBERS);
    let mut joined = None;
    let mut group = c.benchmark_group("policy_store");
    group.bench_with_input(
        BenchmarkId::new("member_write_listed", MEMBERS),
        &MEMBERS,
        |b, _| b.iter(|| join_or_leave(&store, &joining, &mut joined)),
    );
    group.finish();
}

fn view_build(c: &mut Criterion, fixtures: &mut [Fixture]) {
    for (name, affinity) in [
        ("policy_store/view_build", false),
        ("policy_store/view_build_column", true),
    ] {
        let mut group = c.benchmark_group(name);
        for fixture in fixtures.iter_mut() {
            let n = fixture.n;
            let ask = fixture.ask(affinity);
            // Every form is held to what it must build before it is timed.
            assert_eq!(build(&fixture.store, &ask, &mut fixture.out), n);
            assert_eq!(fixture.out.affinity.len(), if affinity { n } else { 0 });
            assert!(fixture
                .out
                .candidates
                .iter()
                .all(|row| row.health.healthy && row.load.is_some()));
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
                b.iter(|| build(&fixture.store, &ask, &mut fixture.out));
            });
        }
        group.finish();
    }
}

fn load_write(c: &mut Criterion, fixtures: &mut [Fixture]) {
    let mut group = c.benchmark_group("policy_store/load_write");
    for fixture in fixtures.iter_mut() {
        let n = fixture.n;
        // Inside the window: the record is folded and waits, so the write
        // is timed without a publication.
        let at = fixture.window.get() + Duration::from_millis(1);
        let mut turn = 0u64;
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                turn += 1;
                let index = (turn as usize).wrapping_mul(67) % n;
                fixture.store.record(
                    black_box(fixture.ids[index]),
                    0,
                    record_of(index, turn),
                    at,
                    LoadSource::Stream,
                );
            });
        });
        // A cell the filter left out wrote nothing, and nothing waits.
        assert!(turn == 0 || fixture.store.publish_due().is_some());
    }
    group.finish();
}

/// One window's publication of the pool: `policy_store/publish/<n>` with no
/// reader holding a world, and `policy_store_held/publish/<n>` beside a
/// reader that keeps the worlds of the last two publications. The older of
/// the two lists the rows the publication would write over, so every
/// publication of the held cell leaves them and takes other rows.
fn publish(c: &mut Criterion, fixtures: &mut [Fixture]) {
    for (group, reader) in [
        ("policy_store/publish", false),
        ("policy_store_held/publish", true),
    ] {
        let mut group = c.benchmark_group(group);
        for fixture in fixtures.iter_mut() {
            let n = fixture.n;
            let mut turn = 0u64;
            let mut kept = [None, None];
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
                b.iter_batched(
                    || {
                        // One record folded inside the window: the pool has
                        // something to publish at the window's end.
                        turn += 1;
                        let index = (turn as usize).wrapping_mul(67) % n;
                        let opened = fixture.window.get();
                        fixture.store.record(
                            fixture.ids[index],
                            0,
                            record_of(index, turn),
                            opened + Duration::from_millis(1),
                            LoadSource::Stream,
                        );
                        if reader {
                            // The last publication's world takes the place
                            // of the one three publications back.
                            kept[(turn % 2) as usize] = Some(fixture.store.world());
                        }
                        let due = opened + PUBLISH_WINDOW;
                        fixture.window.set(due);
                        due
                    },
                    |due| fixture.store.publish(black_box(due)),
                    BatchSize::PerIteration,
                );
            });
            // A cell the filter left out published nothing: the write
            // cell's records may still wait.
            assert!(turn == 0 || fixture.store.publish_due().is_none());
        }
        group.finish();
    }
}

/// What the bench thread asks of the writer's thread.
#[derive(Default)]
struct Remote {
    /// Write at the busy fleet's rate, through the store's own window.
    busy: AtomicBool,
    /// Publications asked for, one record each, and made.
    asked: AtomicU64,
    made: AtomicU64,
    stop: AtomicBool,
}

impl Remote {
    /// One more publication, made on the writer's core before this returns.
    fn publish_one(&self) {
        let target = self.asked.fetch_add(1, Ordering::Release) + 1;
        while self.made.load(Ordering::Acquire) < target {
            std::hint::spin_loop();
        }
    }
}

/// The writer's thread: it spins on its core and writes what it is asked.
/// Busy, it writes ten records per member per second at the host's clock,
/// and the store publishes by its window. Asked for a publication, it
/// writes one record a window after the last, which publishes by itself.
fn spawn_writer(
    remote: Arc<Remote>,
    store: Arc<MemoryStore>,
    ids: Vec<WorkerId>,
    opened: Instant,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let n = ids.len();
        let period = Duration::from_secs(1) / (10 * n as u32);
        let mut turn = 0u64;
        let mut made = 0u64;
        // The asked publications run on a clock of their own, ahead of any
        // instant the host's clock reaches during the run.
        let mut window = opened + Duration::from_secs(86_400);
        let mut next = Instant::now();
        while !remote.stop.load(Ordering::Relaxed) {
            if made < remote.asked.load(Ordering::Acquire) {
                turn += 1;
                let index = (turn as usize).wrapping_mul(67) % n;
                window += PUBLISH_WINDOW;
                store.record(
                    ids[index],
                    0,
                    record_of(index, turn),
                    window,
                    LoadSource::Stream,
                );
                made += 1;
                remote.made.store(made, Ordering::Release);
            } else if remote.busy.load(Ordering::Relaxed) {
                let now = Instant::now();
                if now >= next {
                    turn += 1;
                    let index = (turn as usize).wrapping_mul(67) % n;
                    store.record(
                        ids[index],
                        0,
                        record_of(index, turn),
                        now,
                        LoadSource::Stream,
                    );
                    next = now.max(next + period);
                }
            } else {
                next = Instant::now();
                std::hint::spin_loop();
            }
        }
    })
}

/// Hold the busy cell to what it claims: with the writer busy, a build
/// reads a load that `before` does not hold, which only a publication of
/// one of the writer's records shows. The store publishes within a window
/// of a write: two seconds without one is a broken cell.
fn await_publication(
    store: &MemoryStore,
    ask: &Ask,
    out: &mut ViewBuffers,
    before: &[Option<LoadRow>],
) {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        build(store, ask, out);
        if out
            .candidates
            .iter()
            .map(|row| row.load)
            .ne(before.iter().copied())
        {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "the busy writer's records reached no build within two seconds"
        );
        std::hint::spin_loop();
    }
}

fn cadence(c: &mut Criterion, fixtures: &mut [Fixture]) {
    let mut group = c.benchmark_group("policy_store_cadence");
    for fixture in fixtures.iter_mut() {
        let n = fixture.n;
        // A store of its own, read into the shared buffers: the cells
        // before these moved the shared store's window ahead of the host's
        // clock, and a record the busy writer stamps with that clock would
        // wait there for a window that does not end.
        let (store, ids, opened) = pool_of(n);
        let ask = ask_at(opened, false);
        let remote = Arc::new(Remote::default());
        // The thread starts with the first cell that runs: a filter that
        // selects none of them starts none.
        let writer = Cell::new(None);
        let start = || {
            if let Some(started) = writer.take() {
                writer.set(Some(started));
            } else {
                writer.set(Some(spawn_writer(
                    Arc::clone(&remote),
                    Arc::clone(&store),
                    ids.clone(),
                    opened,
                )));
            }
        };
        let out = &mut fixture.out;
        group.throughput(Throughput::Elements(n as u64));
        for state in ["quiet", "busy"] {
            let ran = Cell::new(false);
            group.bench_with_input(BenchmarkId::new(state, n), &n, |b, _| {
                start();
                ran.set(true);
                remote.busy.store(state == "busy", Ordering::Relaxed);
                b.iter(|| build(&store, &ask, out));
                remote.busy.store(false, Ordering::Relaxed);
            });
            if state == "busy" && ran.get() {
                build(&store, &ask, out);
                let before: Vec<Option<LoadRow>> =
                    out.candidates.iter().map(|row| row.load).collect();
                remote.busy.store(true, Ordering::Relaxed);
                await_publication(&store, &ask, out, &before);
                remote.busy.store(false, Ordering::Relaxed);
            }
        }
        group.bench_with_input(BenchmarkId::new("after_publish", n), &n, |b, _| {
            start();
            b.iter_batched(
                || remote.publish_one(),
                |()| build(&store, &ask, out),
                BatchSize::PerIteration,
            );
        });
        remote.stop.store(true, Ordering::Relaxed);
        if let Some(writer) = writer.take() {
            writer.join().expect("the writer's thread ends when asked");
        }
    }
    group.finish();
}

/// One membership write of a fleet of pairs: `n` prefills and `n` decodes
/// of one model, every one ready with a record on file and published, none
/// naming a runtime. `policy_store_pairs/kept/<n>` writes one prefill's
/// registration again as it stands, which moves no offer;
/// `policy_store_pairs/moved/<n>` names its runtime and unnames it in turn,
/// which moves its offer at every write. A store of its own a size: the
/// other cells' pools hold one role.
fn pair_writes(c: &mut Criterion) {
    for n in SIZES {
        let facts_of = |index: usize| {
            let (role, key) = if index < n {
                (Role::Prefill, format!("http://p{index}:8000"))
            } else {
                (Role::Decode, format!("http://d{}:8000", index - n))
            };
            WorkerFacts {
                key: key.clone(),
                engine: key,
                incarnation: 1,
                role,
                models: vec![MODEL.to_string()],
                dp_size: 1,
                max_running: Some(4),
                ..WorkerFacts::default()
            }
        };
        let store = MemoryStore::new();
        let opened = Instant::now();
        for index in 0..2 * n {
            let id = store.upsert(facts_of(index)).id();
            let revision = WorkersStore::get(&store, id).expect("the row").revision;
            store
                .set_status(id, WorkerStatus::Ready, revision)
                .expect("ready");
            store.record(id, 0, record_of(index, 0), opened, LoadSource::Stream);
        }
        store.publish(opened + PUBLISH_WINDOW);
        assert_eq!(store.publish_due(), None);
        let written = facts_of(n / 2);
        let named = WorkerFacts {
            engine_kind: "rt-b".to_string(),
            ..written.clone()
        };
        let mut group = c.benchmark_group("policy_store_pairs");
        group.bench_with_input(BenchmarkId::new("kept", n), &n, |b, _| {
            b.iter_batched(
                || written.clone(),
                |facts| store.upsert(black_box(facts)),
                BatchSize::SmallInput,
            );
        });
        // The writes come in the order their facts were made: named,
        // unnamed, named, so each moves the offer the one before left.
        let mut name = false;
        group.bench_with_input(BenchmarkId::new("moved", n), &n, |b, _| {
            b.iter_batched(
                || {
                    name = !name;
                    if name {
                        named.clone()
                    } else {
                        written.clone()
                    }
                },
                |facts| store.upsert(black_box(facts)),
                BatchSize::SmallInput,
            );
        });
        group.finish();
    }
}

/// One membership write beside a worker that lists many models, in a store
/// of its own a size: eight workers of one model and one worker that lists
/// `ids` models of its own, every one ready with a record on file and
/// published. `policy_store_models/member_write/<ids>` writes one of the
/// eight's registration again as it stands. The write reads every row's
/// models into their pools: the row fails when that read looks back over
/// the models a row listed before.
fn member_write_beside_many_models(c: &mut Criterion) {
    let facts_of = |key: String, models: Vec<String>| WorkerFacts {
        engine: key.clone(),
        key,
        incarnation: 1,
        models,
        dp_size: 1,
        max_running: Some(4),
        ..WorkerFacts::default()
    };
    let small = |index: usize| facts_of(format!("http://w{index}:8000"), vec![MODEL.to_string()]);
    let mut group = c.benchmark_group("policy_store_models");
    for ids in [128_usize, 1_024, 4_096] {
        let listed = (0..ids).map(|model| format!("model-{model}")).collect();
        let wide = facts_of("http://wide:8000".to_string(), listed);
        let store = MemoryStore::new();
        let opened = Instant::now();
        for (index, facts) in (0..8).map(&small).chain([wide]).enumerate() {
            let id = store.upsert(facts).id();
            let revision = WorkersStore::get(&store, id).expect("the row").revision;
            store
                .set_status(id, WorkerStatus::Ready, revision)
                .expect("ready");
            store.record(id, 0, record_of(index, 0), opened, LoadSource::Stream);
        }
        store.publish(opened + PUBLISH_WINDOW);
        assert_eq!(store.publish_due(), None);
        let written = small(3);
        group.bench_with_input(BenchmarkId::new("member_write", ids), &ids, |b, _| {
            b.iter_batched(
                || written.clone(),
                |facts| store.upsert(black_box(facts)),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn store_view(c: &mut Criterion) {
    // Every size's buffers are placed before the first store is built. A
    // count's runner bounds the program break, and the stores cross that
    // bound: a buffer asked for after them is mapped on its own and never
    // starts a page.
    let placed: Vec<ViewBuffers> = SIZES
        .iter()
        .map(|&n| ViewBuffers {
            candidates: page_start_buffer(n),
            affinity: page_start_buffer(n),
            stalled: Vec::new(),
            ..ViewBuffers::default()
        })
        .collect();
    let mut fixtures: Vec<Fixture> = SIZES
        .iter()
        .zip(placed)
        .map(|(&n, out)| Fixture::new(n, out))
        .collect();
    floor(c, &mut fixtures);
    view_build(c, &mut fixtures);
    load_write(c, &mut fixtures);
    publish(c, &mut fixtures);
    cadence(c, &mut fixtures);
    vetoed(c, &mut fixtures);
    writes_beside_vetoes(c, &mut fixtures);
    member_write_beside_spread_vetoes(c);
    member_write_under_lists(c);
    pair_writes(c);
    member_write_beside_many_models(c);
}

criterion_group!(benches, store_view);
criterion_main!(benches);
