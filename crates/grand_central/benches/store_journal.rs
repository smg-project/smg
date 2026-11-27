//! What the replication journal costs the store, at 8, 64 and 512 members
//! of one pool, each cell beside its twin on a store with no journal, in
//! one binary:
//!
//! - `policy_store_journal/status_write/<plain|journaled|told>/<n>`: one
//!   status flip of a member, which publishes nothing; `told` is the journaled
//!   write on a log with a listener of the watched kinds, as a controller
//!   with a watch open has: the write's telling is in the cell;
//! - `policy_store_journal/member_write/<plain|journaled>/<n>`: one refresh
//!   of a member's facts, the pool derived again;
//! - `policy_store_journal/read/<events>`: one reader's copy of a batch of
//!   status events out of the log, each with the models its row serves, the
//!   longest the log's lock is held;
//! - `policy_store_journal/read/watched_at_head`: a watch's read at the
//!   log's head: nothing to copy, the lock and the look at the log alone.
//!   The log of the three read cells holds a full ring;
//! - `policy_store_journal/cut/watched/<n>`: one cut of the rows: the copy
//!   a writer of the rows waits for, and the sort after it.
//!
//! Every member is ready, in a store that leads a term as a serving
//! controller's does. The journaled store's log retains 4,096 events.
//!
//! The stores have no holder of worker objects. Each write is a decided
//! change handed to the store ([`MemoryStore::follow`]): the bench states
//! the row and counts the epoch. A store that leads its term journals such
//! a write as its own.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer"
)]

#[path = "common/filled.rs"]
mod filled;

use std::{hint::black_box, sync::Arc, time::Instant};

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion,
};
use smg_grand_central::{
    engine::{Role, WorkerId},
    service::replication::Log,
    store::{
        identity, LeasesStore, MembershipChange, MemoryStore, Position, Registrant, WorkerFacts,
        WorkerRow, WorkerStatus, WorkersStore,
    },
};

const SIZES: [usize; 3] = [8, 64, 512];
const MODEL: &str = "m";
/// The log's ring.
const WATCHED: usize = 4_096;

fn facts(index: usize, priority: i32) -> WorkerFacts {
    let key = format!("http://w{index}:8000");
    WorkerFacts {
        key: key.clone(),
        engine: key,
        incarnation: 1,
        models: vec![MODEL.to_string()],
        dp_size: 1,
        max_running: Some(4),
        priority,
        ..WorkerFacts::default()
    }
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

/// The status of the row `id` moved from `from` to `to`, as the store's
/// write `epoch`.
fn moved(store: &MemoryStore, epoch: u64, id: WorkerId, from: WorkerStatus, to: WorkerStatus) {
    let change = MembershipChange::StatusChanged { id, from, to };
    store.follow(Position::default(), epoch, change, None);
}

/// The store stands at the epoch the bench counted: each write landed.
fn landed(store: &MemoryStore, epoch: u64) {
    assert_eq!(WorkersStore::epoch(store), epoch, "a write did not land");
}

/// One pool of `n` members, each ready and published, in a store with no
/// journal or with a log: the store, its log and its members.
struct Pool {
    store: MemoryStore,
    log: Option<Arc<Log>>,
    ids: Vec<WorkerId>,
}

fn pool_of(n: usize, journaled: bool) -> Pool {
    pool_on(n, journaled.then(|| Log::new(WATCHED)))
}

/// How a cell builds its pool of so many members.
type PoolOf = fn(usize) -> Pool;

/// [`pool_of`] on `log`.
fn pool_on(n: usize, log: Option<Log>) -> Pool {
    let log = log.map(Arc::new);
    let store = match &log {
        Some(log) => MemoryStore::journaled(Arc::clone(log) as _),
        None => MemoryStore::new(),
    };
    // The store leads: its writes are journaled under its term.
    store.set_term(1);
    let now = Instant::now();
    let mut epoch = WorkersStore::epoch(&store);
    let ids: Vec<WorkerId> = (0..n)
        .map(|index| {
            let row = row_of(facts(index, 0), WorkerStatus::Pending, 1, now);
            let id = row.id;
            let registered = MembershipChange::Registered { id };
            store.follow(Position::default(), epoch + 1, registered, Some(row));
            store.publish_now();
            epoch += 2;
            moved(
                &store,
                epoch,
                id,
                WorkerStatus::Pending,
                WorkerStatus::Ready,
            );
            id
        })
        .collect();
    landed(&store, epoch);
    filled::filled(&store, MODEL, Role::Single, n);
    Pool { store, log, ids }
}

const KINDS: [(&str, bool); 2] = [("plain", false), ("journaled", true)];

fn status_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_journal/status_write");
    let pools: [(&str, PoolOf); 3] = [
        ("plain", |n| pool_of(n, false)),
        ("journaled", |n| pool_of(n, true)),
        // The listener does nothing: the cell holds what the log's telling
        // adds to the write, and none of a watch's work.
        ("told", |n| {
            pool_on(n, Some(Log::new(WATCHED).on_watched(|| {})))
        }),
    ];
    for (kind, build) in pools {
        for n in SIZES {
            flips(&mut group, kind, n, &build(n));
        }
    }
    group.finish();
}

/// One status flip of the pool's middle member: no world is published.
fn flips(group: &mut BenchmarkGroup<'_, WallTime>, kind: &str, n: usize, pool: &Pool) {
    let id = pool.ids[n / 2];
    let (mut from, mut to) = (WorkerStatus::Ready, WorkerStatus::NotReady);
    let mut epoch = WorkersStore::epoch(&pool.store);
    group.bench_with_input(BenchmarkId::new(kind, n), &n, |b, _| {
        b.iter(|| {
            epoch += 1;
            moved(&pool.store, epoch, black_box(id), from, to);
            (from, to) = (to, from);
        });
    });
    landed(&pool.store, epoch);
}

fn member_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_journal/member_write");
    group.sample_size(30);
    for (kind, journaled) in KINDS {
        for n in SIZES {
            let pool = pool_of(n, journaled);
            let id = pool.ids[n / 2];
            let now = Instant::now();
            let mut turn = 0i32;
            let mut epoch = WorkersStore::epoch(&pool.store);
            let mut revision = 1;
            group.bench_with_input(BenchmarkId::new(kind, n), &n, |b, _| {
                b.iter(|| {
                    turn = turn.wrapping_add(1);
                    epoch += 1;
                    revision += 1;
                    // The member's row as its refresh leaves it.
                    let row = row_of(facts(n / 2, turn), WorkerStatus::Ready, revision, now);
                    let refreshed = MembershipChange::Relabelled { id };
                    let store = &pool.store;
                    store.follow(Position::default(), epoch, refreshed, Some(black_box(row)));
                    store.publish_now();
                });
            });
            landed(&pool.store, epoch);
        }
    }
    group.finish();
}

fn read(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_journal/read");
    let pool = pool_of(64, true);
    let log = pool.log.as_ref().expect("the journaled pool's log");
    // The ring full and written half over again, as a controller
    // that ran a while holds it: the ring's buffer is wrapped at its
    // middle, and a reader's search of it is the search of a full ring.
    let mut epoch = WorkersStore::epoch(&pool.store);
    for turn in 0..WATCHED + WATCHED / 2 {
        let (from, to) = if turn % 2 == 0 {
            (WorkerStatus::Ready, WorkerStatus::NotReady)
        } else {
            (WorkerStatus::NotReady, WorkerStatus::Ready)
        };
        epoch += 1;
        moved(&pool.store, epoch, pool.ids[0], from, to);
    }
    landed(&pool.store, epoch);
    // The store took term 1 before its first write: every position stands
    // under it.
    let mut after = Position::default();
    after.seq = log.head().seq - 2_048;
    after.term = 1;
    for events in [16usize, 256] {
        let mut out = Vec::with_capacity(events);
        // Checked ahead of the timed id: a filtered run skips its routine.
        log.read(after, events, &mut out).expect("retained");
        assert_eq!(out.len(), events);
        group.bench_with_input(BenchmarkId::from_parameter(events), &events, |b, _| {
            b.iter(|| {
                log.read(black_box(after), events, &mut out)
                    .expect("retained");
                black_box(out.len())
            });
        });
    }
    let head = log.head();
    let mut out = Vec::with_capacity(16);
    log.read(head, 16, &mut out).expect("retained");
    assert!(out.is_empty(), "the reader is at the head");
    group.bench_function("watched_at_head", |b| {
        b.iter(|| {
            log.read(black_box(head), 16, &mut out).expect("retained");
            black_box(out.len())
        });
    });
    group.finish();
}

fn cut(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_journal/cut");
    for n in SIZES {
        let pool = pool_of(n, true);
        group.bench_with_input(BenchmarkId::new("watched", n), &n, |b, _| {
            b.iter(|| black_box(pool.store.checkpoint()));
        });
    }
    group.finish();
}

criterion_group!(benches, status_write, member_write, read, cut);
criterion_main!(benches);
