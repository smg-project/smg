//! The cost of the store's writes, at 8, 64 and 512 members of one pool:
//!
//! - `policy_store/floor/<n>`: the decision benches' gated build, rows
//!   written from facts in hand, in this binary: a cell that asks the store
//!   nothing, so one job reads it beside the store's writes;
//! - `policy_store/veto_write/512` and `policy_store/member_write/512`: one
//!   veto's write and one membership write beside 512 standing vetoes of
//!   eight other gateways;
//! - `policy_store/member_write_spread/512`: one membership write beside
//!   6,144 standing vetoes of twelve other gateways, the 512 members spread
//!   over 64 pools;
//! - `policy_store/member_write_listed/512`: one membership write under a
//!   generation that declares the view's three lists: the worker written
//!   is read under them and what the 512 members read there is carried.
//!
//! This binary is built on the base branch with the pull request's bench
//! sources, so a row here calls nothing the base's store lacks.
//!
//! The stores have no holder of worker objects. Each membership write is a
//! decided change handed to the store ([`MemoryStore::follow`]): the bench
//! states the row a registration leaves and counts the epoch. The
//! registration rule is the holder's, and no cell here times it.
//!
//! `policy_store_labels/member_write/512` is the listed membership write
//! under a generation that declares the label selectors and no family and
//! no zone: no member stands outside such lists, and the write's build
//! reads none under them. The gate leaves it out.
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
//! Every member is ready. The floor's rows are written into a buffer on a
//! page's start (`placed`), one placement per size for the whole binary:
//! the helper walks its earlier rejects first, and a size placed a fifth
//! time stops at its bound.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer"
)]

#[expect(
    dead_code,
    reason = "the shared fleet's engine builder belongs to the decision benches"
)]
mod common;
#[path = "common/filled.rs"]
mod filled;
#[path = "common/placed.rs"]
mod placed;

use std::{
    hint::black_box,
    sync::Arc,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use placed::page_start_buffer;
use smg_grand_central::{
    document::{AppliedSpec, PolicySpec},
    engine::{Candidate, Role, WorkerId},
    store::{
        identity, GatewayId, HealthSink, MembershipChange, MemoryStore, Position, Registrant,
        WorkerFacts, WorkerRow, WorkerStatus, WorkersStore,
    },
};

const SIZES: [usize; 3] = [8, 64, 512];
const MODEL: &str = "m";

/// One pool of `n` members in a store, and the buffer the floor's rows are
/// timed in.
struct Fixture {
    n: usize,
    store: Arc<MemoryStore>,
    ids: Vec<WorkerId>,
    rows: Vec<Candidate>,
}

/// The row of `facts` as a registration by a control call leaves it, in
/// `status` at `revision`. The stores here have no holder of worker
/// objects: each takes a row as a decided change ([`MemoryStore::follow`]).
/// The bench states the row and counts the epoch the write steps to.
fn row_of(
    facts: WorkerFacts,
    status: WorkerStatus,
    revision: u64,
    admitted_at: Instant,
) -> WorkerRow {
    WorkerRow {
        id: identity(&facts.key, facts.incarnation),
        facts,
        status,
        revision,
        admitted_at,
        maker: Registrant::Control,
    }
}

/// One member registered and ready in `store`, as its next two writes:
/// its row, with the build the row owes paid, and its status.
fn ready_member(store: &MemoryStore, facts: WorkerFacts) -> WorkerId {
    let row = row_of(facts, WorkerStatus::Pending, 1, Instant::now());
    let id = row.id;
    let epoch = WorkersStore::epoch(store) + 1;
    let registered = MembershipChange::Registered { id };
    store.follow(Position::default(), epoch, registered, Some(row));
    store.publish_now();
    let ready = MembershipChange::StatusChanged {
        id,
        from: WorkerStatus::Pending,
        to: WorkerStatus::Ready,
    };
    store.follow(Position::default(), epoch + 1, ready, None);
    landed(store, epoch + 1);
    id
}

/// The store stands at the epoch the bench counted: each write landed.
fn landed(store: &MemoryStore, epoch: u64) {
    assert_eq!(WorkersStore::epoch(store), epoch, "a write did not land");
}

/// One row written again in place as the store's write `epoch`, and the
/// build it leaves owed.
fn written_again(store: &MemoryStore, epoch: u64, row: WorkerRow) {
    let refreshed = MembershipChange::Relabelled { id: row.id };
    store.follow(Position::default(), epoch, refreshed, Some(row));
    store.publish_now();
}

/// A store with one pool of `n` members, every one registered, ready and
/// published: the store and its members by index.
fn pool_of(n: usize) -> (Arc<MemoryStore>, Vec<WorkerId>) {
    let store = Arc::new(MemoryStore::new());
    let ids: Vec<WorkerId> = (0..n)
        .map(|index| {
            let key = format!("http://w{index}:8000");
            let facts = WorkerFacts {
                key: key.clone(),
                engine: key,
                incarnation: 1,
                models: vec![MODEL.to_string()],
                dp_size: 1,
                max_running: Some(4),
                ..WorkerFacts::default()
            };
            ready_member(&store, facts)
        })
        .collect();
    assert_eq!(store.publish_due(), None);
    filled::filled(&store, &ids, MODEL);
    (store, ids)
}

impl Fixture {
    fn new(n: usize, rows: Vec<Candidate>) -> Self {
        let (store, ids) = pool_of(n);
        Self {
            n,
            store,
            ids,
            rows,
        }
    }
}

/// The gated build's floor in this binary: the rows written from facts in
/// hand (`engine/view_build`'s own build over the same synthetic
/// fleet) into the fixture's buffer, so one run on one machine reads the
/// floor beside the store's writes.
fn floor(c: &mut Criterion, fixtures: &mut [Fixture]) {
    let mut group = c.benchmark_group("policy_store/floor");
    for fixture in fixtures.iter_mut() {
        let n = fixture.n;
        let fleet = common::fleet(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                common::build_view(black_box(&fleet), &mut fixture.rows);
                black_box(&fixture.rows);
                fixture.rows.len()
            });
        });
    }
    group.finish();
}

/// A worker that joins a pool and leaves it in turn: whether it is in,
/// and the store's epoch as the bench counts it.
struct Turn {
    joined: bool,
    epoch: u64,
}

impl Turn {
    fn of(store: &MemoryStore) -> Self {
        Self {
            joined: false,
            epoch: WorkersStore::epoch(store),
        }
    }
}

/// One membership write in turn, and the build it leaves owed: the worker
/// joins where it is out and leaves where it is in. `joining` is its row
/// as a registration leaves it.
fn join_or_leave(store: &MemoryStore, joining: &WorkerRow, turn: &mut Turn) {
    let id = joining.id;
    turn.epoch += 1;
    if turn.joined {
        let left = MembershipChange::Removed { id };
        store.follow(Position::default(), turn.epoch, left, None);
    } else {
        let registered = MembershipChange::Registered { id };
        let row = black_box(joining.clone());
        store.follow(Position::default(), turn.epoch, registered, Some(row));
    }
    turn.joined = !turn.joined;
    store.publish_now();
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
    // The gateway whose veto is written, and a veto's end.
    let asking = GatewayId(1);
    let until = Instant::now() + Duration::from_secs(3_600);
    let others = (2..10).map(GatewayId);
    let vetoed = |nth: usize| fixture.ids[nth * n / 64];
    for gateway in others.clone() {
        for nth in 0..64 {
            store.gateway_veto(gateway, vetoed(nth), Some(until));
        }
    }
    let mut group = c.benchmark_group("policy_store");
    // A member the others' vetoes name as well: the asking gateway's entry
    // stands beside theirs on one worker.
    let target = vetoed(31);
    let mut set = false;
    group.bench_with_input(BenchmarkId::new("veto_write", n), &n, |b, _| {
        b.iter(|| {
            set = !set;
            store.gateway_veto(asking, black_box(target), set.then_some(until));
        });
    });
    store.gateway_veto(asking, target, None);
    let joining = WorkerFacts {
        key: "http://joining:8000".to_string(),
        engine: "http://joining:8000".to_string(),
        incarnation: 1,
        models: vec![MODEL.to_string()],
        dp_size: 1,
        max_running: Some(4),
        ..WorkerFacts::default()
    };
    let joining = row_of(joining, WorkerStatus::Pending, 1, Instant::now());
    let mut turn = Turn::of(store);
    group.bench_with_input(BenchmarkId::new("member_write", n), &n, |b, _| {
        b.iter(|| join_or_leave(store, &joining, &mut turn));
    });
    group.finish();
    if turn.joined {
        join_or_leave(store, &joining, &mut turn);
    }
    // Each write of the turn landed, and a turn that wrote ended on a
    // leave: the joining member is out, and the pool is as it was.
    landed(store, turn.epoch);
    for gateway in others {
        for nth in 0..64 {
            store.gateway_veto(gateway, vetoed(nth), None);
        }
    }
    filled::filled(store, &fixture.ids, MODEL);
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
            let facts = facts(format!("http://s{index}:8000"), index % POOLS);
            ready_member(&store, facts)
        })
        .collect();
    assert_eq!(store.publish_due(), None);
    let spread: Vec<WorkerId> = ids.iter().copied().step_by(POOLS).collect();
    filled::filled(&store, &spread, "spread-0");
    let until = Instant::now() + Duration::from_secs(3_600);
    for gateway in (2..14).map(GatewayId) {
        for id in &ids {
            store.gateway_veto(gateway, *id, Some(until));
        }
    }
    let joining = facts("http://joining-spread:8000".to_string(), 0);
    let joining = row_of(joining, WorkerStatus::Pending, 1, Instant::now());
    let mut turn = Turn::of(&store);
    let mut group = c.benchmark_group("policy_store");
    group.bench_with_input(
        BenchmarkId::new("member_write_spread", MEMBERS),
        &MEMBERS,
        |b, _| b.iter(|| join_or_leave(&store, &joining, &mut turn)),
    );
    group.finish();
    landed(&store, turn.epoch);
}

/// `policy_store/member_write_listed/<n>` is one membership write under a
/// generation that declares the view's three lists, in a store of its own:
/// 512 members of one pool, each stating an engine family, a zone and
/// twelve labels, under two families, four zones and eight selectors. A
/// worker joins the pool and leaves it in turn: the write's build reads
/// that worker under the lists and carries what every other member read.
fn member_write_under_lists(c: &mut Criterion) {
    member_write_under(c, "policy_store", "member_write_listed", true);
}

/// `policy_store_labels/member_write/<n>` is that write under a generation
/// that declares the eight selectors and no family and no zone, over the
/// same members. No member stands outside such lists: the write's build
/// reads no member under them, as under no list.
fn member_write_under_labels_alone(c: &mut Criterion) {
    member_write_under(c, "policy_store_labels", "member_write", false);
}

/// One membership write over 512 members of one pool under a generation's
/// lists, as the cell `cell` of `group`: the eight selectors, and with
/// `places` the two families and the four zones.
fn member_write_under(c: &mut Criterion, group: &str, cell: &str, places: bool) {
    const MEMBERS: usize = 512;
    let store = MemoryStore::new();
    let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
    document.generation = 1;
    if places {
        document.view.engines = vec!["alpha".to_string(), "beta".to_string()];
        document.view.zones = (0..4).map(|zone| format!("z{zone}")).collect();
    }
    document.view.labels = (0..8).map(|key| format!("k{key}")).collect();
    let listing = AppliedSpec::validate(&document).expect("the document validates");
    store.apply(listing).expect("applied");
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
    let ids: Vec<WorkerId> = (0..MEMBERS)
        .map(|index| ready_member(&store, facts(format!("http://l{index}:8000"), index)))
        .collect();
    assert_eq!(store.publish_due(), None);
    filled::filled(&store, &ids, MODEL);
    // A fifth of the members state a zone the list of zones does not
    // name: where that list stands, the store counts those members
    // outside, and under the selectors alone it counts none.
    assert_eq!(store.outside_lists().zones > 0, places);
    let joining = facts("http://joining-listed:8000".to_string(), MEMBERS);
    let joining = row_of(joining, WorkerStatus::Pending, 1, Instant::now());
    let mut turn = Turn::of(&store);
    let mut group = c.benchmark_group(group);
    group.bench_with_input(BenchmarkId::new(cell, MEMBERS), &MEMBERS, |b, _| {
        b.iter(|| join_or_leave(&store, &joining, &mut turn));
    });
    group.finish();
    landed(&store, turn.epoch);
}

/// One membership write of a fleet of pairs: `n` prefills and `n` decodes
/// of one model, every one ready and published, none naming a runtime. `policy_store_pairs/kept/<n>` writes one prefill's
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
        let ids: Vec<WorkerId> = (0..2 * n)
            .map(|index| ready_member(&store, facts_of(index)))
            .collect();
        assert_eq!(store.publish_due(), None);
        let (prefills, decodes) = ids.split_at(n);
        filled::filled(&store, prefills, MODEL);
        filled::filled(&store, decodes, MODEL);
        let written = facts_of(n / 2);
        let named = WorkerFacts {
            engine_kind: "rt-b".to_string(),
            ..written.clone()
        };
        // Each write is the prefill's row as a refresh leaves it: ready,
        // one revision on.
        let now = Instant::now();
        let (mut epoch, mut revision) = (WorkersStore::epoch(&store), 1);
        let mut group = c.benchmark_group("policy_store_pairs");
        group.bench_with_input(BenchmarkId::new("kept", n), &n, |b, _| {
            b.iter_batched(
                || {
                    revision += 1;
                    row_of(written.clone(), WorkerStatus::Ready, revision, now)
                },
                |row| {
                    epoch += 1;
                    written_again(&store, epoch, black_box(row));
                },
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
                    revision += 1;
                    let facts = if name { &named } else { &written };
                    row_of(facts.clone(), WorkerStatus::Ready, revision, now)
                },
                |row| {
                    epoch += 1;
                    written_again(&store, epoch, black_box(row));
                },
                BatchSize::SmallInput,
            );
        });
        group.finish();
        landed(&store, epoch);
    }
}

/// One membership write beside a worker that lists many models, in a store
/// of its own a size: eight workers of one model and one worker that lists
/// `ids` models of its own, every one ready and published.
/// `policy_store_models/member_write/<ids>` writes one of the eight's
/// registration again as it stands. The write reads every row's models into
/// their pools: the row fails when that read looks back over the models a
/// row listed before.
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
        let facts = (0..8).map(&small).chain([wide]);
        let members: Vec<WorkerId> = facts.map(|facts| ready_member(&store, facts)).collect();
        assert_eq!(store.publish_due(), None);
        let (small_ones, wide_one) = members.split_at(8);
        filled::filled(&store, small_ones, MODEL);
        filled::filled(&store, wide_one, "model-0");
        let written = small(3);
        let now = Instant::now();
        let (mut epoch, mut revision) = (WorkersStore::epoch(&store), 1);
        group.bench_with_input(BenchmarkId::new("member_write", ids), &ids, |b, _| {
            b.iter_batched(
                || {
                    revision += 1;
                    row_of(written.clone(), WorkerStatus::Ready, revision, now)
                },
                |row| {
                    epoch += 1;
                    written_again(&store, epoch, black_box(row));
                },
                BatchSize::SmallInput,
            );
        });
        landed(&store, epoch);
    }
    group.finish();
}

fn store_view(c: &mut Criterion) {
    // Every size's buffer is placed before the first store is built. A
    // count's runner bounds the program break, and the stores cross that
    // bound: a buffer asked for after them is mapped on its own and never
    // starts a page.
    let placed: Vec<Vec<Candidate>> = SIZES.iter().map(|&n| page_start_buffer(n)).collect();
    let mut fixtures: Vec<Fixture> = SIZES
        .iter()
        .zip(placed)
        .map(|(&n, rows)| Fixture::new(n, rows))
        .collect();
    floor(c, &mut fixtures);
    writes_beside_vetoes(c, &mut fixtures);
    member_write_beside_spread_vetoes(c);
    member_write_under_lists(c);
    member_write_under_labels_alone(c);
    pair_writes(c);
    member_write_beside_many_models(c);
}

criterion_group!(benches, store_view);
criterion_main!(benches);
